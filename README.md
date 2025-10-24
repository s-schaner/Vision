# VolleySense CalibMap

VolleySense CalibMap is a calibration-first playground for volleyball analytics. The web app lets you upload a match video, calibrate the camera-to-court homography, and orchestrate LLM-powered tracking for the ball and players. An optional Python sidecar adds OpenCV + YOLO preprocessing to enrich the prompts.

## Repository layout

```
web/                # Static client (open index.html directly)
  index.html
  app.js
  style.css
  llm_prompts.js
  assets/court.svg
server/
  preprocess.py     # Flask YOLO preprocessing service
  requirements.txt
```

## Running the web client

1. Open `web/index.html` in any modern desktop browser (Chromium, Firefox, Safari).
2. Load a local volleyball video (MP4, MOV, or WebM).
3. Grab a calibration frame and click 4–8 known court points. After entering overlay coordinates (0..1000), confirm the homography to unlock overlays, grid back-projection, and RMS reporting.
4. Use the Ball / Net / Player pickers to annotate anchors (ball radius, ground contact, player bounding boxes + foot points).
5. Configure the analyzer pane (LLM endpoint, sampling fps, resize dimensions) and generate ball or player tracks. Downloads update automatically and load into the court animation.
6. Drag JSON files (ball_track.json or humans.json) onto the court pane to visualise external data.

Keyboard shortcut: press **Space** to toggle video playback.

## Python preprocessing service

1. Install dependencies:
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch the Flask service:
   ```bash
   python preprocess.py
   ```
3. Call `/preprocess` via `curl` or any HTTP client. Example:
   ```bash
   curl -X POST http://localhost:5001/preprocess \
     -F file=@/path/to/volleyball.mp4 \
     -F fps=8 -F resize_w=640 -F resize_h=360 \
     -F H="[[1,0,0],[0,1,0],[0,0,1]]"
   ```
   The response contains frame-by-frame detections projected into overlay space when `H` is provided.

> The service gracefully degrades when YOLO weights are missing, emitting heuristic detections so you can validate the contract without GPU support.

## LLM configuration

* Set the **LLM Base URL** and **Model** fields to match your OpenAI-compatible endpoint.
* The client first tries `POST {baseUrl}/chat/completions`, then falls back to `POST {baseUrl}/responses`.
* Prompts are defined in `web/llm_prompts.js` and enforce JSON-only responses that follow the required schemas.
* When no LLM endpoint is provided, the app falls back to heuristic projections based on calibration anchors.

## Data contracts

### `ball_track.json`
```
{
  "meta": {
    "overlay_frame": "EXTENDED_COURT_+20FT",
    "units": {"x": "0..1000", "y": "0..1000", "z": "m"},
    "fps": 12.5
  },
  "track": [
    {"idx": 0, "t": 0.000, "x": 512.3, "y": 478.1, "z": 1.23}
  ]
}
```

### `humans.json`
```
{
  "meta": {
    "overlay_frame": "EXTENDED_COURT_+20FT",
    "units": {"x": "0..1000", "y": "0..1000"},
    "classes": ["player"]
  },
  "humans": [
    {"idx": 0, "t": 0.000, "list": [{"id": "A", "x": 305.4, "y": 642.1, "conf": 0.92}]}
  ]
}
```

### `calibration.json` (exportable)
```
{
  "H": [[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]],
  "img_points": [{"x": ..., "y": ..., "overlay": {"x": ..., "y": ...}}],
  "net_posts_img": [{"x": ..., "y": ...}, {"x": ..., "y": ...}],
  "net": {"y_overlay": 500.0, "height_m": 2.43},
  "ball_anchor_img": {"x": ..., "y": ...},
  "ball_ref_radius_px": 28.0,
  "player_box_img": {"x": ..., "y": ..., "w": ..., "h": ...},
  "player_foot_img": {"x": ..., "y": ...}
}
```

## Calibration tips

* Capture a frame where court lines are crisp and unobstructed.
* Click at least four non-collinear references (corners, intersection of lines). Enter overlay coordinates in the 0..1000 range following the extended court convention (0,0 = near-left margin).
* Use Shift+Click to remove the nearest calibration point.
* After confirming the homography, inspect the grid overlay and RMS value. Aim for RMS < 5 for accurate reprojection.
* Use the **Pick Net** button to mark the two post bases. Adjust the net height between men (2.43 m) and women (2.24 m) by editing the `netHeight` field in the UI prompt (or modify in code).

## Net height

The default net height is **2.43 m** (men). You can adapt it in the UI prompts when positioning the net or by editing `state.calibration.netHeight` inside `web/app.js` (search for `netHeightMen`).

Happy calibrating and volley-analysing!
