# NeuroEar

A browser app that listens to your microphone, recognizes spoken words with a pretrained
convolutional neural network, and shows how the network processes the sound, layer by layer.

```
Microphone ─► Spectrogram ─► Neural network (3D, live activations) ─► Softmax output
```

Stack: Vite, vanilla JS, Three.js, `@tensorflow/tfjs` 3.x, `@tensorflow-models/speech-commands`
(the `BROWSER_FFT` recognizer). It recognizes 18 words: zero–nine, up, down, left, right, go, stop,
yes and no. It also has two extra classes, `_background_noise_` and `_unknown_`.

## Run it

```bash
npm install
npm run dev
```

Open **http://localhost:5173** in Chrome or Firefox, then click **Enable microphone**.
Browsers only allow microphone access on `localhost` or `https`.

**Using it:** press **Record** (or Space), say one word, then press **Recognize** (or Space again).
Recording stops, and the word goes through the pipeline. Recording stops on its own after 3 s.

For the presentation you can also serve the production build:

```bash
npm run build
npm run preview        # http://localhost:4173
```

## Offline setup

Everything is bundled locally, so no CDN is used:

- **Libraries and fonts:** these come from npm and are bundled by Vite (Space Grotesk and Inter
  come from `@fontsource`).
- **The model:** it lives in `public/model/` (included in the repo):

  | file | what |
  |---|---|
  | `model.json` | the network's architecture and the weights manifest |
  | `group1-shard1of2`, `group1-shard2of2` | the weights (about 5.6 MB) |
  | `metadata.json` | the 20 output labels, in order |

  The app loads it with
  `speechCommands.create('BROWSER_FFT', undefined, '/model/model.json', '/model/metadata.json')`.
  To download it again (you must be online), run `npm run setup-model`. Add `-- --force` to
  overwrite the existing files.

**Check before you present:** run `npm run dev` once, turn off Wi‑Fi, reload the page and say a word.

## Before the presentation

1. **Calibrate** (button in the header, or **C**). Do this on the presentation laptop and, if you
   can, in the room. The panel asks you to say each of the 18 words and 6 non-words, using
   Space to record and Space again to recognize. It then shows how many words were accepted,
   rejected or confused with the current thresholds, and suggests better ones. Among settings that
   score equally, the suggestion picks the strictest, so it never loosens the thresholds without a
   gain. **Apply suggested** saves the thresholds in this browser; **Reset to defaults** undoes it.
2. **Record backup demo clips** in case the microphone or the room lets you down. Run
   `npm run dev`, record a word, then click **💾 Save as demo clip**. The clip is saved as
   `public/demo/<word>.wav`. Saved clips appear under *backup clips* in the microphone panel.
   Press **1–9** (or click a clip) to play it out loud and run it through exactly the same
   pipeline as a live recording. Saving and deleting clips only works with `npm run dev`; a built
   app (`npm run build`) just plays the clips that are in `public/demo`. A clip of a non-word,
   such as "banana", is a good way to show the network answering *unknown*.
3. **Warm-up** happens on its own. While the start screen says *Warming up the neural network…*,
   the model runs once on dummy data, so the GPU code is compiled before your first live word
   rather than during it.

## Keyboard shortcuts (presenter mode)

| key | action |
|---|---|
| **Space** / **Enter** | Record, then Recognize (stops the recording and processes the word) |
| **→ / ←** | Step through the pipeline one stage at a time (after a result) |
| **Esc** | Leave step mode |
| **F** | Toggle fullscreen |
| **R** | Reset the history |
| **H** | Hide or show the captions |
| **D** | Toggle the 2D debug view of every layer, with the verification result |
| **1–9** | Play backup demo clip 1–9 |
| **C** | Open or close the calibration panel |

Mouse controls in the 3D view: drag to rotate, scroll to zoom, hover over a layer to see its name,
type and output shape, and double-click to reset the camera.

The steps are generated from the model's real layers: Microphone → Spectrogram → each Conv,
Pool, Flatten, Dropout and Dense layer → Softmax → Prediction. Each step has a one-line caption.

## How it works

1. **Recording.** An AudioWorklet captures raw 44.1 kHz microphone samples on the audio thread,
   so no audio is dropped even when the page is busy rendering. The spectrogram frames are
   computed from those samples with the same features that speech-commands'
   `BrowserFftFeatureExtractor` reads from its AnalyserNode:
   - a Blackman window over 2048 samples, an FFT, then |X|/N in dB, with no smoothing;
   - one frame every 1024 samples, keeping the first 232 bins (≈ 0–5 kHz).

   This was checked against a real AnalyserNode and matches to within 0.001 dB (`src/spectrum.js`).
2. **Finding the word.** The network hears exactly 1 second (43 frames). The app finds the
   speech in the recording: loud stretches separated by less than 0.2 s count as one utterance,
   and the loudest utterance wins. The 1-second window is centered on it and outlined in orange on
   the waveform and the spectrogram. The library's own `recognizer.recognize()` classifies it.
3. **Saying "unknown".** The model always picks one of its 20 classes and is often over-confident
   on words it has never heard. A command word is only accepted when **all** of these hold
   (`ACCEPT` in `src/recognizer.js`):
   - the network's top class is a command word, with ≥ 90 % probability;
   - the same word stays at ≥ 75 % on average when the window is shifted by ±70 ms and ±140 ms
     (real words are stable; false matches usually flip);
   - the speech lasts no longer than 1.2 s (longer means a phrase, not a command word).

   Otherwise the result is **unknown** or **noise**, and the app shows the reason. In tests with
   synthesized speech (2 voices), this cut false accepts of non-command words from 44/44 to 5/44,
   with no command word ever reported as a *different* command. It can't be perfect: words that
   sound like a command (tree → "three", green → "three") still get through, because the network
   is 100 % sure and stays sure when the window is shifted.
4. **Activations.** The recognizer's `tf.LayersModel` is `recognizer.model`. The app falls back to
   `tf.loadLayersModel` if that property is missing. `model.summary()` is printed to the console. An
   activation model, `tf.model({ inputs, outputs: model.layers.map(l => l.output) })`, receives the
   same 1-second window that the recognizer classified. The window is z-score normalized exactly
   as `browser_fft_utils.normalize()` does it: `(x − mean) / (√variance + ε)`.
5. **Verification.** The output of the last layer is compared with the scores from `recognizer.recognize()`. The header badge
   shows the result, for example `✓ activations verified (Δ 0e+0)`, and so does the console.
6. **Visualization.** The visualization is built from each layer's class name and output shape,
   nothing is hard-coded:
   - 4D outputs (Conv2D, MaxPooling2D) become columns of feature-map tiles. Each layer shows at most
     its 8 most active channels, downsampled to 43 × 64.
   - Dense and Flatten layers become spheres. At most 128 are shown per layer; when a layer is
     larger, each sphere is the max of a group of neurons.
   - Dropout appears as a dashed "gate", because it does nothing at inference.

   All tensor work runs inside `tf.tidy`, and the downloaded arrays are disposed immediately.

## Troubleshooting

- **"Microphone access was blocked":** click the lock or mic icon in the address bar, allow the
  microphone, then press *Try again*.
- **"No audio detected":** check that the mic isn't muted, and pick the right input device in
  Ubuntu Settings → Sound → Input.
- **"Could not load the speech model":** `public/model/` is missing files. Run `npm run setup-model`.
- **`npm run dev` fails with "vite: not found":** the folder path contains `:` (for example
  `Local Disk (D:)`), which breaks npm's PATH handling. The scripts already call
  `node node_modules/vite/bin/vite.js` directly to avoid this. For the same reason,
  `vite.config.js` sets `server.fs.strict = false`.
