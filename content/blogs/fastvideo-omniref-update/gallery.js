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
  const inspectProfile = (panel, listen) => {
    [composition.value, duration.value] = panel.dataset.omniProfile.split(":");
    if (listen) audio.value = listen;
    selectProfile();
    panel.querySelectorAll("details").forEach((details) => { details.open = true; });
    status.textContent = "Selected example: three seed 0 outputs, reference inputs and full prompt below. Playback starts only when you press play.";
    panel.querySelector("h4").focus({ preventScroll: true });
    gallery.scrollIntoView({ block: "start" });
  };
  const inspectHash = () => {
    const panel = panels.find((candidate) => `#${candidate.id}` === window.location.hash);
    if (panel) inspectProfile(panel);
    return Boolean(panel);
  };
  document.querySelectorAll("a[data-omni-open]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const panel = panels.find((candidate) => candidate.dataset.omniProfile === link.dataset.omniOpen);
      if (!panel) return;
      event.preventDefault();
      if (window.location.hash !== link.hash) window.history.pushState(null, "", link.hash);
      inspectProfile(panel, link.dataset.omniListen);
    });
  });
  window.addEventListener("hashchange", inspectHash);
  const browseProfile = () => {
    selectProfile();
    window.history.replaceState(null, "", `#${activePanel().id}`);
  };
  composition.addEventListener("change", browseProfile);
  duration.addEventListener("change", browseProfile);
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
  inspectHash();
})();
