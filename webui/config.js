// Optional Nix overlay. If absent/empty, model-classifier.js uses built-in defaults.
window.__MODEL_CLASSIFIER_CONFIG =
  window.__MODEL_CLASSIFIER_CONFIG || window.__MODEL_ROUTER_CONFIG || null;
// Pre-rename global, kept so an overlay written before the rename still loads.
window.__MODEL_ROUTER_CONFIG = window.__MODEL_CLASSIFIER_CONFIG;
