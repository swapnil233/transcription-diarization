const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const { spawn } = require("child_process");

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: true,
      contextIsolation: false,
    },
  });

  mainWindow.loadFile("index.html");
}

app.whenReady().then(() => {
  createWindow();

  app.on("activate", function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", function () {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.on("open-file-dialog", (event) => {
  dialog
    .showOpenDialog({
      properties: ["openFile"],
      filters: [{ name: "Movies", extensions: ["mkv", "avi", "mp4"] }],
    })
    .then((result) => {
      if (!result.canceled) {
        console.log(result.filePaths[0]);
        const videoPath = `"${result.filePaths[0]}"`; // Use the chosen file path
        const audioPath = `"${result.filePaths[0].replace(
          /\.[^/.]+$/,
          ""
        )}.wav"`; // Create audioPath based on videoPath

        console.log("Spawning ffmpeg");

        const ffmpeg = spawn(
          "ffmpeg",
          [
            "-y",
            "-i",
            videoPath, // Use videoPath directly, without extra quotes
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audioPath,
          ],
          { shell: true }
        );

        console.log("Spawned ffmpeg");

        ffmpeg.stdout.on("data", (data) => {
          console.log(`FFMPEG stdout: ${data.toString()}`);
        });

        ffmpeg.stderr.on("data", (data) => {
          console.error(`FFMPEG stderr: ${data.toString()}`);
        });

        ffmpeg.on("exit", (code) => {
          if (code === 0) {
            const python = spawn("python", ["transcribe.py", audioPath]);
            python.stdout.on("data", (data) => {
              console.log(`stdout: ${data.toString()}`);
            });

            python.stderr.on("data", (data) => {
              console.error(`stderr: ${data.toString()}`);
            });
          }
        });
      }
    })
    .catch((err) => {
      console.log(err);
    });
});
