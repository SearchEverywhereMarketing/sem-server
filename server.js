'use strict';

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const os = require('os');

const execAsync = promisify(exec);
const app = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '500mb' }));
app.use(express.urlencoded({ extended: true, limit: '500mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── POST /api/score  (Anthropic proxy) ───────────────────────────────────────
app.post('/api/score', async (req, res) => {
  console.log('[score] Request received, body size:', JSON.stringify(req.body).length, 'bytes');

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'ANTHROPIC_API_KEY is not configured' });
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify(req.body),
    });

    const data = await response.json();
    res.status(response.status).json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    res.status(502).json({ error: 'Failed to reach Anthropic API', details: message });
  }
});

// ── Shared video processing helpers ──────────────────────────────────────────
async function getVideoDuration(videoPath) {
  console.log('[score-video] Getting duration for:', videoPath);
  const { stdout } = await execAsync(
    `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${videoPath}"`
  );
  const dur = parseFloat(stdout.trim());
  console.log('[score-video] Duration:', dur);
  return dur;
}

async function extractFrames(videoPath, framesDir) {
  console.log('[score-video] Extracting frames to:', framesDir);
  const { stderr } = await execAsync(
    `ffmpeg -i "${videoPath}" -vf fps=1 "${framesDir}/frame_%04d.jpg" -y`
  );
  console.log('[score-video] ffmpeg frames done, stderr tail:', stderr.slice(-300));
}

async function extractAudio(videoPath, audioPath) {
  console.log('[score-video] Extracting audio to:', audioPath);
  const { stderr } = await execAsync(
    `ffmpeg -i "${videoPath}" -ar 16000 -ac 1 -f wav "${audioPath}" -y`
  );
  console.log('[score-video] ffmpeg audio done, stderr tail:', stderr.slice(-300));
}

async function transcribeAudio(audioPath) {
  console.log('[score-video] Transcribing audio:', audioPath);
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY is not configured');

  const audioBuffer = await fsPromises.readFile(audioPath);
  const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });

  const formData = new FormData();
  formData.append('file', audioBlob, 'audio.wav');
  formData.append('model', 'whisper-1');
  formData.append('response_format', 'verbose_json');
  formData.append('timestamp_granularities[]', 'word');

  const response = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: formData,
  });

  console.log('[score-video] Whisper response status:', response.status);

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Whisper API error ${response.status}: ${err}`);
  }

  const data = await response.json();
  console.log('[score-video] Transcript length:', data.text?.length, 'words:', data.words?.length);
  return {
    text: data.text ?? '',
    words: (data.words ?? []).map((w) => ({ word: w.word, start: w.start, end: w.end })),
  };
}

function sampleFrames(frames, max) {
  if (frames.length <= max) return frames;
  const result = [];
  for (let i = 0; i < max; i++) {
    const idx = Math.round((i / (max - 1)) * (frames.length - 1));
    result.push(frames[idx]);
  }
  return result;
}

async function processVideoFile(videoPath, res) {
  const framesDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), 'frames-'));
  const audioPath = path.join(os.tmpdir(), `audio-${Date.now()}.wav`);

  try {
    const duration = await getVideoDuration(videoPath);

    await Promise.all([
      extractFrames(videoPath, framesDir),
      extractAudio(videoPath, audioPath),
    ]);

    const allFrameFiles = (await fsPromises.readdir(framesDir))
      .filter((f) => f.endsWith('.jpg'))
      .sort();

    console.log('[score-video] Total frames extracted:', allFrameFiles.length);

    const sampled = sampleFrames(allFrameFiles, 60);
    console.log('[score-video] Frames after sampling:', sampled.length);

    const frames = await Promise.all(
      sampled.map(async (filename) => {
        const match = filename.match(/frame_(\d+)\.jpg$/);
        const frameNumber = match ? parseInt(match[1], 10) : 0;
        const timestamp = frameNumber - 1;
        const data = await fsPromises.readFile(path.join(framesDir, filename));
        return { b64: data.toString('base64'), timestamp };
      })
    );

    console.log('[score-video] Frames encoded:', frames.length, '— starting transcription');
    const { text: transcript, words } = await transcribeAudio(audioPath);

    console.log('[score-video] Done. Sending response.');
    res.json({ frames, transcript, words, duration });
  } finally {
    await Promise.allSettled([
      fsPromises.rm(framesDir, { recursive: true, force: true }),
      fsPromises.rm(audioPath, { force: true }),
    ]);
    console.log('[score-video] Temp files cleaned up');
  }
}

// ── POST /api/score-video  (video → frames + Whisper transcript) ──────────────
// Accepts EITHER:
//   A) multipart/form-data with a 'video' field (legacy)
//   B) application/json with { videoB64, fileName, mimeType }
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: 500 * 1024 * 1024 },
});

app.post('/api/score-video', (req, res, next) => {
  // Route based on content type
  if (req.is('application/json')) {
    next(); // skip multer, go to JSON handler below
  } else {
    upload.single('video')(req, res, next); // multer handles multipart
  }
}, async (req, res) => {
  console.log('[score-video] Request received, content-type:', req.headers['content-type']);

  let videoPath = null;
  let tempCreated = false;

  try {
    if (req.is('application/json')) {
      // ── JSON base64 path ──────────────────────────────────
      const { videoB64, fileName, mimeType } = req.body;

      if (!videoB64) {
        return res.status(400).json({ error: 'No video data provided (videoB64 missing)' });
      }

      console.log('[score-video] JSON path — fileName:', fileName, 'mimeType:', mimeType, 'b64 length:', videoB64.length);

      // Determine file extension from mimeType or fileName
      const ext = (fileName && path.extname(fileName)) ||
                  (mimeType === 'video/quicktime' ? '.mov' :
                   mimeType === 'video/webm' ? '.webm' :
                   mimeType === 'video/x-matroska' ? '.mkv' : '.mp4');

      videoPath = path.join(os.tmpdir(), `upload-${Date.now()}${ext}`);
      const videoBuffer = Buffer.from(videoB64, 'base64');
      await fsPromises.writeFile(videoPath, videoBuffer);
      tempCreated = true;

      console.log('[score-video] Base64 decoded and written to:', videoPath, 'size:', videoBuffer.length, 'bytes');

    } else {
      // ── Multipart FormData path ───────────────────────────
      const file = req.file;
      if (!file) {
        return res.status(400).json({ error: 'No video file provided' });
      }
      console.log('[score-video] FormData path — file:', file.originalname, 'size:', file.size);
      videoPath = file.path;
    }

    await processVideoFile(videoPath, res);

  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('[score-video] ERROR:', message);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to process video', details: message });
    }
  } finally {
    if (videoPath && tempCreated) {
      await fsPromises.rm(videoPath, { force: true }).catch(() => {});
    } else if (videoPath && req.file) {
      await fsPromises.rm(videoPath, { force: true }).catch(() => {});
    }
  }
});

// ── Global error handler ──────────────────────────────────────────────────────
app.use((err, _req, res, _next) => {
  const status = err.status ?? 500;
  const message = err.message ?? 'Internal server error';
  console.error('[error-handler]', message);
  if (!res.headersSent) {
    res.status(status).json({ error: message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
const server = app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

server.setTimeout(120000);
