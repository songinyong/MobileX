const express = require('express');
const Stream = require('node-rtsp-stream');
// Set up express server
const app = express();
app.use(express.static('public')); // Serve static files from public directory

// Start express server
const server = app.listen(3000, '0.0.0.0', () => console.log('Server started on port 3000'));

// Set up websocket server

console.log('Client connected');

let stream = new Stream({
    name: 'stream',
    streamUrl: 'rtsp://localhost:8554/ds-test',
    wsPort: 9999,
    ffmpegOptions: {
      '-stats': '',
      '-r': 30,
      '-crf': 18
    }
  });
