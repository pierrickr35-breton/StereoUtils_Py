# StereoUtils_Py

Stereonets and paleomagnetic utilities: plotting and statistics of directional data, VGPs and poles, rotations, predicted geomagnetic field curves, maps.

## Download (macOS standalone application)

**[Download the latest version](https://github.com/pierrickr35-breton/StereoUtils_Py/releases/latest)** — on the release page, under *Assets*, click `StereoUtils_Py.zip`.

- This is a **standalone macOS application**: Python and all libraries are included, there is nothing else to install.
- It runs on Macs with an **Apple Silicon** processor (M1 and later) only. On an Intel Mac, Windows or Linux, run the program from the source code (see below).
- Do **not** use the "Source code (zip)" / "Source code (tar.gz)" links of the release page unless you want the Python source: they do not contain the application.

### Installation

1. Double-click `StereoUtils_Py.zip` to unzip it, and move `StereoUtils_Py.app` to your *Applications* folder.
2. The application is not signed with an Apple developer certificate, so macOS blocks it the first time. Try to open it once, then go to *System Settings > Privacy & Security*, scroll down and click **Open Anyway**. This is needed only once.

   If macOS says the application "is damaged and can't be opened", open the Terminal and type:

   ```bash
   xattr -dr com.apple.quarantine /Applications/StereoUtils_Py.app
   ```

## User Guide

In the application: *Help > User Guide*. The same guide is in this repository: `help/StereoUtils_Py_Guide.html`.

## Running from the source code (any platform)

Requires Python 3 with Tkinter, and: numpy matplotlib scipy pmagpy cartopy ppigrf certifi.

```bash
pip install numpy matplotlib scipy pmagpy cartopy ppigrf certifi
python3 app.py
```
