(() => {
  "use strict";
  const chartSelect = document.querySelector("[data-omni-chart-duration]");
  if (chartSelect) {
    chartSelect.addEventListener("change", () => {
      document.querySelectorAll("[data-omni-chart]").forEach((chart) => {
        chart.hidden = chart.dataset.omniChart !== chartSelect.value;
      });
    });
  }
  const gallery = document.querySelector("[data-omni-gallery]");
  if (!gallery) return;
  const composition = gallery.querySelector("[data-omni-composition]");
  const duration = gallery.querySelector("[data-omni-duration]");
  const audio = gallery.querySelector("[data-omni-audio]");
  const status = gallery.querySelector("[data-omni-status]");
  const panels = Array.from(gallery.querySelectorAll("[data-omni-profile]"));
  const activePanel = () => panels.find((panel) => !panel.hidden);
  const outputs = () => Array.from(activePanel().querySelectorAll(".omni-output video"));
  const pauseAll = () => gallery.querySelectorAll("video,audio").forEach((media) => media.pause());
  const selectAudio = () => outputs().forEach((video) => {
    video.muted = video.dataset.omniVariant !== audio.value;
  });
  const selectProfile = () => {
    pauseAll();
    const key = `${composition.value}:${duration.value}`;
    panels.forEach((panel) => { panel.hidden = panel.dataset.omniProfile !== key; });
    selectAudio();
    status.textContent = "Matched seed 0 clips. Select Play together or use each player's controls.";
  };
  composition.addEventListener("change", selectProfile);
  duration.addEventListener("change", selectProfile);
  audio.addEventListener("change", selectAudio);
  gallery.querySelector("[data-omni-pause]").addEventListener("click", () => {
    pauseAll();
    status.textContent = "Playback paused.";
  });
  gallery.querySelector("[data-omni-play]").addEventListener("click", async () => {
    pauseAll();
    selectAudio();
    try {
      await Promise.all(outputs().map((video) => {
        video.currentTime = 0;
        return video.play();
      }));
      status.textContent = "Playing all three clips from the beginning. Audio follows Listen to.";
    } catch (_) {
      status.textContent = "Use each player's controls if your browser blocks synchronized playback.";
    }
  });
  selectProfile();
})();
