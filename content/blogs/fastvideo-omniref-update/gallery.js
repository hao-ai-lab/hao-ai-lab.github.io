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
  let playbackRequest = 0;
  const pauseAll = () => {
    playbackRequest += 1;
    gallery.querySelectorAll("video,audio").forEach((media) => media.pause());
  };
  const selectAudio = () => outputs().forEach((video) => {
    video.muted = video.dataset.omniVariant !== audio.value;
  });
  const selectProfile = () => {
    pauseAll();
    const key = `${composition.value}:${duration.value}`;
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.omniProfile !== key;
      // Every example shows references immediately and starts with a compact prompt.
      panel.querySelectorAll("details").forEach((detail) => {
        detail.open = detail.classList.contains("omni-input-details");
      });
    });
    selectAudio();
    if (chartSelect) {
      chartSelect.value = duration.value;
      document.querySelectorAll("[data-omni-chart]").forEach((chart) => { chart.hidden = chart.dataset.omniChart !== duration.value; });
    }
    status.textContent = `${activePanel().querySelector("h3").textContent}. Ready to compare.`;
  };
  const inspectProfile = (panel, listen) => {
    [composition.value, duration.value] = panel.dataset.omniProfile.split(":");
    if (listen) audio.value = listen;
    selectProfile();
    // Deep links keep references visible and leave the full prompt collapsed.
    panel.querySelector("h3").focus({ preventScroll: true });
    gallery.scrollIntoView({ block: "start", behavior: "instant" });
  };
  const inspectHash = () => {
    const panel = panels.find((candidate) => `#${candidate.id}` === window.location.hash);
    if (panel) { inspectProfile(panel); return; }
    // Preserve links to steps and headings nested in collapsed technical sections.
    let target;
    try { target = document.getElementById(decodeURIComponent(window.location.hash.slice(1))); }
    catch (_) { return; }
    if (!target) return;
    for (let ancestor = target; ancestor; ancestor = ancestor.parentElement) {
      if (ancestor.tagName === "DETAILS") ancestor.open = true;
    }
    target.scrollIntoView({ block: "start", behavior: "instant" });
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
  gallery.querySelector("[data-omni-next]").addEventListener("click", () => {
    composition.selectedIndex = (composition.selectedIndex + 1) % composition.options.length;
    browseProfile();
  });
  if (chartSelect) chartSelect.addEventListener("change", () => {
    duration.value = chartSelect.value;
    browseProfile();
  });
  gallery.querySelector("[data-omni-pause]").addEventListener("click", () => {
    pauseAll();
    status.textContent = "All clips paused.";
  });
  gallery.querySelector("[data-omni-play]").addEventListener("click", async () => {
    pauseAll();
    const request = playbackRequest;
    const videos = outputs();
    selectAudio();
    status.textContent = `Loading ${videos.length} clips…`;
    videos.forEach((video) => { video.currentTime = 0; });
    const results = await Promise.allSettled(videos.map((video) => video.play()));
    if (request !== playbackRequest) return;
    if (results.some((result) => result.status === "rejected")) {
      pauseAll();
      status.textContent = "Playback could not start for all clips. Try each player’s play control.";
    } else {
      status.textContent = `Playing ${videos.length} clips. Audio follows Listen to.`;
    }
  });
  gallery.querySelectorAll("[data-omni-copy]").forEach((button) => {
    if (!navigator.clipboard?.writeText) return;
    button.hidden = false;
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(button.nextElementSibling.textContent);
        button.textContent = "Copied";
        status.textContent = "Full prompt copied.";
      } catch (_) {
        status.textContent = "Unable to copy automatically. Select the prompt text and copy it.";
      }
    });
  });
  selectProfile();
  inspectHash();
})();
