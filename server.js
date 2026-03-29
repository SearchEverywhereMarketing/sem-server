'use strict';

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const ffmpegStatic = require('ffmpeg-static');
const ffmpeg = require('fluent-ffmpeg');
const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs');
const fsPromises = require('fs/promises');
const path = require('path');
const os = require('os');

// Point fluent-ffmpeg at the static binary
ffmpeg.setFfmpegPath(ffmpegStatic);

// Also set ffprobe path — ffmpeg-static includes ffprobe
const ffprobePath = ffmpegStatic.replace('ffmpeg', 'ffprobe');
ffmpeg.setFfprobePath(ffprobePath);

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
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err, metadata) => {
      if (err) return reject(err);
      const dur = metadata.format.duration || 0;
      console.log('[score-video] Duration:', dur);
      resolve(dur);
    });
  });
}

async function extractFrames(videoPath, framesDir) {
  console.log('[score-video] Extracting frames to:', framesDir);
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions(['-vf', 'fps=1'])
      .output(path.join(framesDir, 'frame_%04d.jpg'))
      .on('end', () => {
        console.log('[score-video] ffmpeg frames done');
        resolve();
      })
      .on('error', (err) => {
        console.error('[score-video] ffmpeg frames error:', err.message);
        reject(err);
      })
      .run();
  });
}

async function extractAudio(videoPath, audioPath) {
  console.log('[score-video] Extracting audio to:', audioPath);
  return new Promise((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions(['-ar', '16000', '-ac', '1', '-f', 'wav'])
      .output(audioPath)
      .on('end', () => {
        console.log('[score-video] ffmpeg audio done');
        resolve();
      })
      .on('error', (err) => {
        console.error('[score-video] ffmpeg audio error:', err.message);
        reject(err);
      })
      .run();
  });
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

    // Transcription is best-effort — frames always returned even if Whisper fails
    let transcript = '';
    let words = [];
    try {
      const result = await transcribeAudio(audioPath);
      transcript = result.text;
      words = result.words;
    } catch (transcriptErr) {
      console.warn('[score-video] Transcription failed (non-fatal):', transcriptErr.message);
    }

    console.log('[score-video] Done. Sending response. transcript length:', transcript.length);
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

// ── POST /api/generate-image  (DALL-E 3 image generation) ────────────────────
app.post('/api/generate-image', async (req, res) => {
  console.log('[generate-image] Request received');
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'OPENAI_API_KEY is not configured' });

  const { prompt, size } = req.body;
  if (!prompt) return res.status(400).json({ error: 'prompt is required' });

  // Map platform to DALL-E size
  const dalleSize = size || '1024x1024';

  try {
    const response = await fetch('https://api.openai.com/v1/images/generations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'dall-e-3',
        prompt,
        n: 1,
        size: dalleSize,
        quality: 'hd',
        response_format: 'b64_json',
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      console.error('[generate-image] DALL-E error:', JSON.stringify(data));
      return res.status(response.status).json({ error: data.error?.message || 'DALL-E error', details: data });
    }

    const b64 = data.data?.[0]?.b64_json;
    const revised_prompt = data.data?.[0]?.revised_prompt;
    console.log('[generate-image] Success, b64 length:', b64?.length);
    res.json({ b64, revised_prompt });
  } catch (err) {
    console.error('[generate-image] ERROR:', err.message);
    res.status(502).json({ error: 'Failed to generate image', details: err.message });
  }
});

// ── POST /api/analyze-image  (GPT-4o vision — read product image + write prompt) ─
app.post('/api/analyze-image', async (req, res) => {
  console.log('[analyze-image] Request received');
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'OPENAI_API_KEY is not configured' });

  const { imageB64, mimeType, brandName, industry, description, platform, outputType } = req.body;
  if (!imageB64) return res.status(400).json({ error: 'imageB64 is required' });

  const platformDimensions = {
    instagram: '1080x1080 square',
    facebook: '1200x630 landscape',
    youtube: '1280x720 landscape thumbnail',
    tiktok: '1080x1920 vertical portrait',
  };

  const systemPrompt = `You are a world-class commercial photographer and art director specialising in food, product, and local business photography. 

Your job is to analyse a product photo and write a detailed DALL-E 3 prompt that:
1. Recreates the same product in a dramatically more professional, editorial setting
2. Preserves the exact product — same item, same colors, same branding visible
3. Creates a compelling scene appropriate for the platform and business type
4. Is optimised for ${platform?.toUpperCase()} at ${platformDimensions[platform] || '1080x1080 square'} dimensions
5. Output type: ${outputType}

Return ONLY a JSON object with this exact structure:
{
  "prompt": "detailed DALL-E 3 generation prompt here",
  "scene_description": "one sentence describing the scene",
  "cta_text": "suggested call-to-action text for overlay, max 6 words"
}`;

  const userMessage = `Business: ${brandName || 'Local Business'}
Industry: ${industry || 'Food & Beverage'}
Platform: ${platform}
Output type: ${outputType}
Brief: ${description || 'Create professional marketing content for this product'}

Analyse this product image and write a DALL-E 3 prompt to recreate it professionally.`;

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: 'gpt-4o',
        max_tokens: 800,
        messages: [
          { role: 'system', content: systemPrompt },
          {
            role: 'user',
            content: [
              {
                type: 'image_url',
                image_url: { url: `data:${mimeType || 'image/jpeg'};base64,${imageB64}` },
              },
              { type: 'text', text: userMessage },
            ],
          },
        ],
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      console.error('[analyze-image] GPT-4o error:', JSON.stringify(data));
      return res.status(response.status).json({ error: data.error?.message || 'GPT-4o error' });
    }

    const raw = data.choices?.[0]?.message?.content || '';
    const clean = raw.replace(/^```json\s*/,'').replace(/\s*```$/,'').trim();
    const parsed = JSON.parse(clean);
    console.log('[analyze-image] Success, prompt length:', parsed.prompt?.length);
    res.json(parsed);
  } catch (err) {
    console.error('[analyze-image] ERROR:', err.message);
    res.status(502).json({ error: 'Failed to analyze image', details: err.message });
  }
});

// ── POST /api/score-image  (Claude vision scoring for generated images) ────────
app.post('/api/score-image', async (req, res) => {
  console.log('[score-image] Request received');
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return res.status(500).json({ error: 'ANTHROPIC_API_KEY is not configured' });

  const { imageB64, mimeType, platform, outputType, brandName, industry } = req.body;
  if (!imageB64) return res.status(400).json({ error: 'imageB64 is required' });

  const systemPrompt = `You are a content quality evaluator specialising in local business marketing visuals.

Score this generated marketing image for a ${brandName || 'local business'} in the ${industry || 'Food & Beverage'} industry.
Platform: ${platform?.toUpperCase()}
Output type: ${outputType}

Score it across these dimensions on a 0-10 scale:
- visual_impact: Does it stop the scroll? Is it striking?
- product_clarity: Is the product clearly the hero? Is it appetising/appealing?
- platform_fit: Does it match ${platform} content norms and dimensions?
- brand_professionalism: Does it look like a professional brand?
- cta_potential: Would this make someone want to buy/visit?
- composition: Is the layout and visual hierarchy strong?

Return ONLY valid JSON:
{
  "overall_score": 0.0,
  "visual_impact": 0,
  "product_clarity": 0,
  "platform_fit": 0,
  "brand_professionalism": 0,
  "cta_potential": 0,
  "composition": 0,
  "passes": true,
  "priority_fix": "one specific fix if score < 8.0",
  "prompt_adjustment": "specific instruction to improve the DALL-E prompt if score < 8.0"
}`;

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 600,
        system: systemPrompt,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: mimeType || 'image/png', data: imageB64 } },
            { type: 'text', text: 'Score this marketing image. Return only valid JSON.' },
          ],
        }],
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      console.error('[score-image] Claude error:', JSON.stringify(data));
      return res.status(response.status).json({ error: 'Claude scoring error', details: data });
    }

    const raw = data.content?.map(b => b.text || '').join('') || '';
    const clean = raw.replace(/^```json\s*/,'').replace(/\s*```$/,'').trim();
    const parsed = JSON.parse(clean);
    parsed.passes = parsed.overall_score >= 8.0;
    console.log('[score-image] Score:', parsed.overall_score, 'passes:', parsed.passes);
    res.json(parsed);
  } catch (err) {
    console.error('[score-image] ERROR:', err.message);
    res.status(502).json({ error: 'Failed to score image', details: err.message });
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
  console.log('[startup] ANTHROPIC_API_KEY present:', !!process.env.ANTHROPIC_API_KEY);
  console.log('[startup] OPENAI_API_KEY present:', !!process.env.OPENAI_API_KEY);
  console.log('[startup] ffmpeg path:', ffmpegStatic);
  console.log('[startup] ffprobe path:', require('ffprobe-static').path);
});

server.setTimeout(120000);
