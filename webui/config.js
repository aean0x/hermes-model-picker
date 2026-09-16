// Optional Nix overlay. If absent/empty, model-picker.js uses built-in defaults.
window.__MODEL_PICKER_CONFIG =
  window.__MODEL_PICKER_CONFIG ||
  window.__MODEL_CLASSIFIER_CONFIG ||
  window.__MODEL_ROUTER_CONFIG ||
  null;
// Pre-rename globals, kept so an overlay written by an older deployment loads.
window.__MODEL_CLASSIFIER_CONFIG = window.__MODEL_PICKER_CONFIG;
window.__MODEL_ROUTER_CONFIG = window.__MODEL_PICKER_CONFIG;
