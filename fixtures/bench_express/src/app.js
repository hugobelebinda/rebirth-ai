/** Tiny Express app shell (customer-like surface). */

import express from 'express';
import { formatDuration } from './duration.js';

export function createApp() {
  const app = express();
  app.get('/health', (_req, res) => {
    res.json({ ok: true, uptime: formatDuration(process.uptime() * 1000) });
  });
  return app;
}
