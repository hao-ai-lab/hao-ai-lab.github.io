(() => {
  "use strict";
  const pauseEveryGallery = () => document.querySelectorAll(".omni-gallery video,.omni-gallery audio")
    .forEach((media) => media.pause());
  const galleries = Array.from(document.querySelectorAll("[data-omni-gallery]")).map((gallery) => {
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
      gallery.querySelectorAll("[data-omni-timing-row]").forEach((row) => {
        row.hidden = row.dataset.omniSeconds !== duration.value;
      });
      gallery.querySelector("[data-omni-table-seconds]").textContent = duration.value;
      selectAudio();
      status.textContent = "Prompt, references and outputs are displayed below. Play together restarts all three clips.";
    };
    const inspect = (panel, listen) => {
      [composition.value, duration.value] = panel.dataset.omniProfile.split(":");
      if (listen) audio.value = listen;
      selectProfile();
      panel.querySelector(".omni-example-title").focus({ preventScroll: true });
      gallery.scrollIntoView({ block: "start" });
    };
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
      pauseEveryGallery();
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
    return { panels, inspect };
  });

  const inspectHash = () => {
    for (const gallery of galleries) {
      const panel = gallery.panels.find((candidate) => `#${candidate.id}` === window.location.hash);
      if (panel) {
        gallery.inspect(panel);
        return;
      }
    }
  };
  document.querySelectorAll("a[data-omni-open]").forEach((link) => {
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      for (const gallery of galleries) {
        const panel = gallery.panels.find((candidate) => candidate.dataset.omniProfile === link.dataset.omniOpen);
        if (!panel) continue;
        event.preventDefault();
        if (window.location.hash !== link.hash) window.history.pushState(null, "", link.hash);
        gallery.inspect(panel, link.dataset.omniListen);
        return;
      }
    });
  });
  window.addEventListener("hashchange", inspectHash);
  inspectHash();
})();
