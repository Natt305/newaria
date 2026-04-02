"""
ComfyUI workflow adapters for AriaBot.

Primary: build_refchain_workflow() — programmatic FLUX.2 Klein workflow using
  ReferenceChainConditioning (ComfyUI-ReferenceChain). One node accepts all
  reference images at once as a JSON list, handles scaling+VAE-encoding internally,
  and chains them as reference latents. Requires one node pack; no JSON file needed.

Multi-character inpainting: build_ultimate_workflow() — uses the "Flux.2 Ultimate
  Inpaint Pro Ultra v3.1" GUI workflow JSON, which supports up to 4 character slots
  with per-character ReferenceLatent conditioning, RMBG background removal,
  InpaintCropImproved per-character inpainting, and optional SAM3 auto-segmentation.
  Activated by setting COMFYUI_MODE=ultimate_inpaint. Requires a ComfyUI server
  with the matching custom node packs installed.

Legacy (kept for reference): AIO per-segment inpainting workflow expander.
  Converts the subgraph-format "Flux.2 AIO Pro Simple v3.1" GUI workflow JSON
  into the flat API format. Requires 6+ custom node packs; no longer used as
  the primary path.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Anatomy quality suffix — must match _ANATOMY_SUFFIX in comfyui_ai.py.
# Duplicated here to avoid a circular import (comfyui_ai imports workflow_adapter).
_ANATOMY_SUFFIX = (
    ", perfect anatomy, correct arms, well-drawn hands, "
    "five fingers, proper limbs, symmetrical body"
)

_AIO_PATH = os.path.join(os.path.dirname(__file__), "workflows", "Flux_2_AIO_Pro_Simple_v3_1.json")
_ULTIMATE_PATH = os.path.join(os.path.dirname(__file__), "workflows", "Flux_2_Ultimate_Inpaint_Pro_Ultra_v3_1.json")

# Subgraph UUIDs used in the AIO workflow
_SG_IMAGE_1  = "a195af2c-3c21-4534-9b25-96a8957e40d4"
_SG_IMAGE_2  = "3cd832dc-4c06-4494-b354-86a12561247e"
_SG_IMAGE_3  = "a85f37d4-6c0b-40c9-9e37-6d0d8d4d72d3"
_SG_IMAGE_4  = "00b3cde9-82af-41c4-98c7-4063064debfe"
_SG_MODEL    = "cd156ce4-8fb7-4776-9cfa-9ebd0aec7f91"
_SG_PROMPT   = "9d273a78-edba-46fa-ada3-f43ccfa7aa18"
_SG_SAMPLING = "881e19c1-1466-4052-8d6b-37ebf8d3fab0"


def _parse_link(lk) -> Tuple[int, int, int, int, int]:
    if isinstance(lk, list):
        return lk[0], lk[1], lk[2], lk[3], lk[4]
    return lk["id"], lk["origin_id"], lk["origin_slot"], lk["target_id"], lk["target_slot"]


def _build_link_index(links):
    by_id: Dict[int, Tuple[int, int]] = {}
    targets: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for lk in links:
        lid, o_n, o_s, t_n, t_s = _parse_link(lk)
        by_id[lid] = (o_n, o_s)
        targets[(t_n, t_s)] = (o_n, o_s)
    return by_id, targets


def _topo_order(nodes, links) -> List[dict]:
    node_ids = {n["id"] for n in nodes}
    deps: Dict[int, set] = {n["id"]: set() for n in nodes}
    for lk in links:
        _, o_n, _, t_n, _ = _parse_link(lk)
        if o_n >= 0 and t_n >= 0 and o_n in node_ids and t_n in node_ids:
            deps[t_n].add(o_n)
    visited, ordered = set(), []
    def visit(nid):
        if nid in visited:
            return
        visited.add(nid)
        for d in deps.get(nid, set()):
            visit(d)
        ordered.append(nid)
    for n in nodes:
        visit(n["id"])
    by_id = {n["id"]: n for n in nodes}
    return [by_id[nid] for nid in ordered if nid in by_id]


def _expand(nodes, links, subgraph_defs, parent_inputs, counter):
    """
    Expand a list of ComfyUI GUI nodes+links into API-format flat nodes.

    parent_inputs: {slot_int -> (str_new_node_id, slot_int)} resolves -10 virtual inputs.
    counter: [int] mutable single-element list used as a shared ID allocator.

    Returns (api_nodes_dict, output_map) where:
      api_nodes_dict: {str_new_id: {class_type, inputs, _meta, _role}} API-ready nodes
      output_map: {output_slot -> (str_new_node_id, slot)} for -20 virtual outputs
    """
    api_nodes: Dict[str, dict] = {}
    link_src, _ = _build_link_index(links)

    topo = _topo_order(nodes, links)
    old_to_new: Dict[int, str] = {}
    sg_out_maps: Dict[int, dict] = {}

    def resolve(src_id, src_slot):
        if src_id == -10:
            return parent_inputs.get(src_slot)
        if src_id in old_to_new:
            return (old_to_new[src_id], src_slot)
        if src_id in sg_out_maps:
            return sg_out_maps[src_id].get(src_slot)
        return None

    for n in topo:
        old_id = n["id"]
        ntype = n.get("type", "")

        if ntype in ("Reroute", "Note", "PrimitiveNode"):
            # GUI-only nodes: Reroute passes its single input straight through;
            # Note and PrimitiveNode carry no runtime meaning.
            if ntype == "Reroute":
                inp_list = n.get("inputs", [])
                if inp_list:
                    lnk = inp_list[0].get("link")
                    if lnk and lnk in link_src:
                        o_n, o_s = link_src[lnk]
                        r = resolve(o_n, o_s)
                        if r:
                            old_to_new[old_id] = r[0]
                            # Reroute always passes slot 0 → map via sg_out_maps
                            sg_out_maps[old_id] = {0: r}
            continue

        if ntype in subgraph_defs:
            sg = subgraph_defs[ntype]
            sg_pi = {}
            for inp_slot, inp in enumerate(n.get("inputs", [])):
                lnk = inp.get("link")
                if lnk and lnk in link_src:
                    o_n, o_s = link_src[lnk]
                    r = resolve(o_n, o_s)
                    if r:
                        sg_pi[inp_slot] = r

            inner_api, inner_out = _expand(
                sg.get("nodes", []),
                sg.get("links", []),
                subgraph_defs,
                sg_pi,
                counter,
            )
            api_nodes.update(inner_api)
            sg_out_maps[old_id] = inner_out
        else:
            nid = counter[0]
            counter[0] += 1
            old_to_new[old_id] = str(nid)

            inputs: Dict[str, Any] = {}
            wv = n.get("widgets_values", [])
            w_idx = 0

            for inp_slot, inp in enumerate(n.get("inputs", [])):
                name = inp.get("name", f"input_{inp_slot}")
                lnk = inp.get("link")
                has_widget = inp.get("widget") is not None

                if lnk and lnk in link_src:
                    o_n, o_s = link_src[lnk]
                    r = resolve(o_n, o_s)
                    if r:
                        inputs[name] = [r[0], r[1]]
                    elif has_widget and w_idx < len(wv):
                        # Link goes to an unconnected subgraph input (-10 with no parent)
                        # — fall back to the stored widget default value.
                        inputs[name] = wv[w_idx]
                    if has_widget:
                        w_idx += 1
                elif has_widget:
                    if w_idx < len(wv):
                        inputs[name] = wv[w_idx]
                    w_idx += 1

            title = n.get("title", "") or ntype
            api_nodes[str(nid)] = {
                "class_type": ntype,
                "inputs": inputs,
                "_meta": {"title": title},
                "_orig_id": old_id,
                "_orig_wv": list(wv),
            }

    out_map: Dict[int, Tuple[str, int]] = {}
    for lk in links:
        lid, o_n, o_s, t_n, t_s = _parse_link(lk)
        if t_n == -20:
            r = resolve(o_n, o_s)
            if r:
                out_map[t_s] = r

    return api_nodes, out_map


def expand_aio_workflow(gui_json: dict) -> Dict[str, dict]:
    """Expand the AIO GUI workflow JSON to a flat ComfyUI API-format dict."""
    subgraph_defs = {
        sg["id"]: sg
        for sg in gui_json.get("definitions", {}).get("subgraphs", [])
    }
    counter = [1000]
    api, _ = _expand(
        gui_json["nodes"],
        gui_json["links"],
        subgraph_defs,
        {},
        counter,
    )
    return api


def _find_nodes_by(api: dict, class_type: str = None, title: str = None) -> List[str]:
    """Return list of API node IDs matching class_type and/or title."""
    result = []
    for nid, node in api.items():
        if class_type and node.get("class_type") != class_type:
            continue
        if title and (node.get("_meta", {}).get("title") or "") != title:
            continue
        result.append(nid)
    return result


def _patch_widget(api: dict, nid: str, widget_name: str, value: Any):
    """Patch a single widget input in an API node."""
    api[nid]["inputs"][widget_name] = value


def populate_aio_workflow(
    api_template: Dict[str, dict],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    overall_prompt: str,
    per_segment_prompt: str,
    ref_images: List[Optional[str]],
    seed: int,
    steps: int = 6,
    width: int = 1280,
    height: int = 720,
) -> Dict[str, dict]:
    """
    Populate an already-expanded AIO workflow template with per-run values.

    ref_images: list of up to 4 ComfyUI filenames (or None to disable that slot).
                e.g. ["char1_upload.png", "char2_upload.png", None, None]

    Returns a copy of api_template with all injected values, ready for /prompt.
    """
    import copy
    api = copy.deepcopy(api_template)

    def patch(class_type, title, widget_name, value, first_only=True):
        nids = _find_nodes_by(api, class_type=class_type, title=title)
        if not nids:
            nids = _find_nodes_by(api, class_type=class_type)
        targets = nids[:1] if first_only else nids
        for nid in targets:
            _patch_widget(api, nid, widget_name, value)
        return bool(targets)

    # --- Model filenames ---
    patch("UNETLoader", None, "unet_name", unet_name)
    patch("CLIPLoader", None, "clip_name", clip_name)
    patch("VAELoader",  None, "vae_name",  vae_name)

    # --- Sampler settings ---
    patch("RandomNoise", None, "noise_seed", seed)
    patch("Flux2Scheduler", None, "steps", steps)

    # --- Prompts ---
    patch("PrimitiveStringMultiline", "Overall prompt",       "value", overall_prompt)
    patch("PrimitiveStringMultiline", "Per segment prompt",   "value", per_segment_prompt)

    # --- Reference image slots (Image 1-4) ---
    # Each slot: LoadImageWithSwitch (title="LoadImageWithSwitch") and PrimitiveString ("Use at part")
    # There are exactly 4 image LoadImageWithSwitch nodes (one per Image N subgraph).
    # We identify them by their _orig_id (568 is the inner node ID in every Image N subgraph).
    img_load_nodes = [
        nid for nid, n in api.items()
        if n.get("class_type") == "LoadImageWithSwitch" and n.get("_orig_id") == 568
    ]
    # Sort by API node ID so they match Image 1, 2, 3, 4 order
    img_load_nodes.sort(key=lambda x: int(x))

    part_str_nodes = [
        nid for nid, n in api.items()
        if n.get("class_type") == "PrimitiveString"
        and (n.get("_meta", {}).get("title") or "") == "Use at part"
    ]
    part_str_nodes.sort(key=lambda x: int(x))

    for i, (load_nid, part_nid) in enumerate(zip(img_load_nodes, part_str_nodes)):
        img_filename = ref_images[i] if i < len(ref_images) else None
        enabled = img_filename is not None
        api[load_nid]["inputs"]["image"] = img_filename or "none.png"
        api[load_nid]["inputs"]["enabled"] = enabled
        # "Use at part" should be the 1-based character index
        api[part_nid]["inputs"]["value"] = str(i + 1)

    # Strip internal metadata keys before sending to ComfyUI
    clean: Dict[str, dict] = {}
    for nid, node in api.items():
        clean[nid] = {
            "class_type": node["class_type"],
            "inputs": node["inputs"],
            "_meta": node["_meta"],
        }
    return clean


def load_expanded_aio() -> Optional[Dict[str, dict]]:
    """Load and expand the AIO workflow from disk. Returns None if file missing."""
    if not os.path.exists(_AIO_PATH):
        print(f"[AIOWorkflow] Workflow file not found: {_AIO_PATH}")
        return None
    try:
        with open(_AIO_PATH, "r", encoding="utf-8") as f:
            gui = json.load(f)
        api = expand_aio_workflow(gui)
        print(f"[AIOWorkflow] Loaded and expanded AIO workflow: {len(api)} nodes")
        return api
    except Exception as exc:
        print(f"[AIOWorkflow] Failed to load/expand workflow: {exc}")
        return None


# Pre-expand at import time (cached in module-level variable)
_EXPANDED_AIO: Optional[Dict[str, dict]] = None


def get_expanded_aio() -> Optional[Dict[str, dict]]:
    global _EXPANDED_AIO
    if _EXPANDED_AIO is None:
        _EXPANDED_AIO = load_expanded_aio()
    return _EXPANDED_AIO


def build_aio_workflow(
    overall_prompt: str,
    per_character_prompts: List[str],
    uploaded_filenames: Dict[str, List[str]],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 6,
    width: int = 1280,
    height: int = 720,
) -> Optional[Dict[str, dict]]:
    """
    High-level entry point: build a populated AIO workflow ready for ComfyUI /prompt.

    uploaded_filenames: {subject_name -> [filename1, filename2, ...]}
                        ordered list of subjects whose reference images have been uploaded.
    per_character_prompts: one prompt per subject (appearance description for the inpaint pass).

    Returns the populated workflow dict, or None if the AIO file is unavailable.

    NOTE: This path is RETIRED in favour of build_refchain_workflow() which needs only one
    node pack (ComfyUI-ReferenceChain) instead of six.  Kept for reference.
    """
    template = get_expanded_aio()
    if template is None:
        return None

    # Build ref_images list: one filename per Image slot (up to 4 subjects, first photo each)
    subjects_ordered = list(uploaded_filenames.keys())
    ref_images: List[Optional[str]] = []
    for subj in subjects_ordered[:4]:
        photos = uploaded_filenames.get(subj, [])
        ref_images.append(photos[0] if photos else None)
    while len(ref_images) < 4:
        ref_images.append(None)

    # Per-segment prompt: one line per subject (truncated to 4)
    per_seg = "\n".join(per_character_prompts[:4])

    return populate_aio_workflow(
        api_template=template,
        unet_name=unet_name,
        vae_name=vae_name,
        clip_name=clip_name,
        overall_prompt=overall_prompt,
        per_segment_prompt=per_seg,
        ref_images=ref_images,
        seed=seed,
        steps=steps,
        width=width,
        height=height,
    )



# ─────────────────────────────────────────────────────────────────────────────
# Ultimate Inpaint Pro Ultra v3.1 — multi-character inpainting workflow
# ─────────────────────────────────────────────────────────────────────────────
# Node discovery constants (inner _orig_id values found in the expanded graph).
# These are the internal IDs used within each subgraph; they stay stable across
# template versions as long as the workflow file is not restructured.
_ULTIMATE_ORIG = {
    "unet":          94,   # UNETLoader (inside model subgraph)
    "clip":          92,   # CLIPLoader
    "vae":            8,   # VAELoader
    "width":         37,   # easy int "Width"
    "height":        38,   # easy int "Height"
    "seed":          15,   # RandomNoise
    "steps":        733,   # PrimitiveInt "Steps"
    "prompt_main":  604,   # PrimitiveStringMultiline "Overall prompt"
    "prompt_seg":   605,   # PrimitiveStringMultiline "Per segment prompt"
    "sam_enable":   637,   # PrimitiveBoolean (enable SAM3)
    "sam_query":    662,   # GODMT_SplitString (SAM3 text query)
    "scene_image":    9,   # LoadImageWithSwitch (outer scene/canvas image)
    "char_image":   568,   # LoadImageWithSwitch ×4 (character reference images)
    "char_part":    574,   # PrimitiveString ×4 ("Use at part" index)
}


def _patch_orig(api: Dict[str, dict], orig_id: int, class_type: str,
                widget_name: str, value: Any) -> bool:
    """Patch the first API node matching orig_id + class_type."""
    for node in api.values():
        if node.get("_orig_id") == orig_id and node.get("class_type") == class_type:
            node["inputs"][widget_name] = value
            return True
    return False


def load_expanded_ultimate() -> Optional[Dict[str, dict]]:
    """Load and expand the Ultimate Inpaint GUI workflow to API format."""
    if not os.path.exists(_ULTIMATE_PATH):
        print(f"[UltimateWF] Workflow file not found: {_ULTIMATE_PATH}")
        return None
    try:
        with open(_ULTIMATE_PATH, "r", encoding="utf-8") as f:
            gui = json.load(f)
        api = expand_aio_workflow(gui)
        print(f"[UltimateWF] Loaded and expanded Ultimate workflow: {len(api)} nodes")
        return api
    except Exception as exc:
        print(f"[UltimateWF] Failed to load/expand workflow: {exc}")
        return None


_EXPANDED_ULTIMATE: Optional[Dict[str, dict]] = None


def get_expanded_ultimate() -> Optional[Dict[str, dict]]:
    """Return the cached expanded Ultimate workflow, loading it if needed."""
    global _EXPANDED_ULTIMATE
    if _EXPANDED_ULTIMATE is None:
        _EXPANDED_ULTIMATE = load_expanded_ultimate()
    return _EXPANDED_ULTIMATE


def populate_ultimate_workflow(
    api_template: Dict[str, dict],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    overall_prompt: str,
    per_segment_prompt: str,
    scene_image_filename: str,
    ref_images: List[Optional[str]],
    seed: int,
    steps: int = 8,
    width: int = 1280,
    height: int = 720,
    sam_query: str = "person",
    use_sam: bool = True,
) -> Dict[str, dict]:
    """
    Populate an expanded Ultimate Inpaint workflow template with per-run values.

    scene_image_filename: ComfyUI filename of the main canvas/scene image.
    ref_images:           Up to 4 ComfyUI filenames for character reference images
                          (or None to leave a slot disabled).

    Returns a clean workflow dict ready for POST /prompt.
    """
    import copy
    api = copy.deepcopy(api_template)
    o = _ULTIMATE_ORIG

    # Model loaders
    # The workflow template uses the standard UNETLoader which validates against the
    # non-GGUF model list (always empty here). Swap it for UnetLoaderGGUF which is
    # what the txt2img workflow uses and correctly finds .gguf files.
    for _node in api.values():
        if _node.get("_orig_id") == o["unet"] and _node.get("class_type") == "UNETLoader":
            _node["class_type"] = "UnetLoaderGGUF"
            _node["inputs"].pop("weight_dtype", None)
            break
    _patch_orig(api, o["unet"],   "UnetLoaderGGUF",        "unet_name",   unet_name)
    _patch_orig(api, o["clip"],   "CLIPLoader",            "clip_name",   clip_name)
    _patch_orig(api, o["vae"],    "VAELoader",             "vae_name",    vae_name)

    # Resolution — easy int nodes inside the model subgraph
    _patch_orig(api, o["width"],  "easy int",              "value",       width)
    _patch_orig(api, o["height"], "easy int",              "value",       height)

    # Sampler settings
    _patch_orig(api, o["seed"],   "RandomNoise",           "noise_seed",  seed)
    _patch_orig(api, o["steps"],  "PrimitiveInt",          "value",       steps)

    # Prompts (inside the main inpaint-loop subgraph)
    _patch_orig(api, o["prompt_main"], "PrimitiveStringMultiline", "value", overall_prompt + _ANATOMY_SUFFIX)
    _patch_orig(api, o["prompt_seg"],  "PrimitiveStringMultiline", "value", per_segment_prompt)

    # Main scene / canvas image
    _patch_orig(api, o["scene_image"], "LoadImageWithSwitch", "image",   scene_image_filename)
    _patch_orig(api, o["scene_image"], "LoadImageWithSwitch", "enabled", True)

    # SAM3 auto-segmentation toggle and query string
    _patch_orig(api, o["sam_enable"], "PrimitiveBoolean",   "value",  use_sam)
    _patch_orig(api, o["sam_query"],  "GODMT_SplitString",  "STRING", sam_query)

    # SAM3Segment runtime parameters — lower threshold so it finds characters in
    # generated scenes, and use Merge mode so an empty result returns a blank tensor
    # (not Python None) which the downstream switch nodes can handle without crashing.
    for _nid, _node in api.items():
        if _node.get("class_type") == "SAM3Segment":
            _node["inputs"]["confidence_threshold"] = 0.15
            _node["inputs"]["output_mode"] = "Merged"

    # Character reference image slots — sorted by API node ID so they correspond
    # to character 1, 2, 3, 4 in top-level node order (outer_ids 2, 3, 4, 7).
    char_load_nids = sorted(
        [nid for nid, n in api.items()
         if n.get("class_type") == "LoadImageWithSwitch" and n.get("_orig_id") == o["char_image"]],
        key=lambda x: int(x),
    )
    char_part_nids = sorted(
        [nid for nid, n in api.items()
         if n.get("class_type") == "PrimitiveString" and n.get("_orig_id") == o["char_part"]],
        key=lambda x: int(x),
    )

    for i, load_nid in enumerate(char_load_nids):
        img_fn = ref_images[i] if i < len(ref_images) else None
        # Disabled slots still require a valid image filename for ComfyUI's validator;
        # use the scene image (always uploaded) as a harmless placeholder.
        api[load_nid]["inputs"]["image"]   = img_fn or scene_image_filename
        api[load_nid]["inputs"]["enabled"] = img_fn is not None

    for i, part_nid in enumerate(char_part_nids):
        api[part_nid]["inputs"]["value"] = str(i + 1)

    # SAM loop None-propagation fix:
    # The easy forLoopStart that drives the SAM3 query loop has no initial_value1,
    # so its accumulator starts as Python None. When SAM finds no segments the switch
    # inside the loop propagates None → GrowMaskWithBlur crashes.
    # Fix: find the InvertMask node (gradient-mask fallback, always a valid tensor)
    # and wire it as the SAM loop's initial accumulator.  That way, if SAM fails on
    # every iteration the loop exits with the gradient mask rather than None.
    _invert_mask_nid: Optional[str] = None
    _godmt_nid: Optional[str] = None
    for _nid, _node in api.items():
        if _node.get("class_type") == "InvertMask":
            _invert_mask_nid = _nid
        if _node.get("class_type") == "GODMT_SplitString":
            _godmt_nid = _nid
    if _invert_mask_nid and _godmt_nid:
        # Find the forLoopStart whose 'total' is linked to GODMT_SplitString output
        for _nid, _node in api.items():
            if _node.get("class_type") == "easy forLoopStart":
                _total = _node["inputs"].get("total")
                if isinstance(_total, list) and _total[0] == _godmt_nid:
                    # This is the SAM query loop — inject gradient mask as initial accumulator
                    if "initial_value1" not in _node["inputs"]:
                        _node["inputs"]["initial_value1"] = [_invert_mask_nid, 0]
                    break

    # InpaintCropImproved added a required 'device_mode' input in newer Impact-Pack
    # versions; inject the default ("auto") if the node doesn't already have it.
    for _node in api.values():
        if _node.get("class_type") == "InpaintCropImproved":
            _node["inputs"].setdefault("device_mode", "gpu (much faster)")

    # Strip internal metadata keys before sending to ComfyUI
    clean: Dict[str, dict] = {}
    for nid, node in api.items():
        entry: Dict[str, Any] = {
            "class_type": node["class_type"],
            "inputs":     node["inputs"],
        }
        if "_meta" in node:
            entry["_meta"] = node["_meta"]
        clean[nid] = entry
    return clean


def build_ultimate_workflow(
    overall_prompt: str,
    per_character_prompts: List[str],
    uploaded_filenames: Dict[str, List[str]],
    scene_image_filename: str,
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 8,
    width: int = 1280,
    height: int = 720,
    sam_query: Optional[str] = None,
    use_sam: bool = True,
) -> Optional[Dict[str, dict]]:
    """
    High-level entry point: build a populated Ultimate Inpaint workflow ready for
    ComfyUI /prompt.

    uploaded_filenames:      {subject_name -> [comfyui_filename, ...]} — first
                             filename per subject is used as the character reference.
    per_character_prompts:   One appearance description per subject (parallel to
                             uploaded_filenames order). Used as the per-segment prompt.
    scene_image_filename:    ComfyUI filename of the main scene/canvas image. Pass
                             a blank white image filename if no pre-existing scene.

    Returns the populated workflow dict, or None if the Ultimate file is unavailable.
    """
    template = get_expanded_ultimate()
    if template is None:
        return None

    subjects_ordered = list(uploaded_filenames.keys())
    ref_images: List[Optional[str]] = []
    for subj in subjects_ordered[:4]:
        photos = uploaded_filenames.get(subj, [])
        ref_images.append(photos[0] if photos else None)
    while len(ref_images) < 4:
        ref_images.append(None)

    per_seg = "\n".join(per_character_prompts[:4]) if per_character_prompts else overall_prompt

    # Build a SAM query string from subject names so SAM3 can find each character.
    if not sam_query and subjects_ordered:
        sam_query = ", ".join(subjects_ordered[:4])
    sam_query = sam_query or "person"

    return populate_ultimate_workflow(
        api_template=template,
        unet_name=unet_name,
        vae_name=vae_name,
        clip_name=clip_name,
        overall_prompt=overall_prompt,
        per_segment_prompt=per_seg,
        scene_image_filename=scene_image_filename,
        ref_images=ref_images,
        seed=seed,
        steps=steps,
        width=width,
        height=height,
        sam_query=sam_query,
        use_sam=use_sam,
    )


def build_refchain_workflow(
    prompt: str,
    subject_filenames: Dict[str, List[str]],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 6,
    width: int = 1280,
    height: int = 720,
) -> Dict[str, dict]:
    """Build a ComfyUI API workflow with one ReferenceChainConditioning node per character.

    Each character gets their own node fed only their own photos, chained in sequence:

      CLIPTextEncode -> ConditioningZeroOut
        -> ReferenceChain[char1] -> ReferenceChain[char2] -> ...
        -> CFGGuider -> SamplerCustomAdvanced -> VAEDecode -> SaveImage

    This prevents the node from cross-referencing photos between characters (the
    previous single-node approach sent all photos at once with no per-character
    isolation, causing outfit bleed).

    Node IDs "6", "7", "8", ... are allocated to ReferenceChainConditioning nodes
    in character order.  Fixed infrastructure nodes use IDs 1-5 and 31-37, 9.

    Args:
        prompt:            Raw text prompt (anatomy suffix appended internally).
        subject_filenames: {character_name: [comfyui_filename, ...]} grouped per
                           character so each node only sees its own photos.
        unet_name:         GGUF model filename for UnetLoaderGGUF.
        vae_name:          VAE filename for VAELoader.
        clip_name:         CLIP filename for CLIPLoader.
        seed:              Random noise seed.
        steps:             Sampler step count.
        width / height:    Output resolution.

    Returns:
        ComfyUI API-format workflow dict, ready for the /prompt endpoint.
    """
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    megapixels = round((width * height) / 1_000_000, 4)

    # Build one ReferenceChainConditioning node per character that has photos.
    # Node IDs start at "6"; each node feeds the next (chain).
    _chars_with_photos = [(name, fnames) for name, fnames in subject_filenames.items() if fnames]
    refchain_nodes: Dict[str, dict] = {}
    _prev_id: Optional[str] = None
    for i, (char_name, fnames) in enumerate(_chars_with_photos):
        node_id = str(6 + i)
        cond_input = [_prev_id, 0] if _prev_id else ["4", 0]
        neg_input  = [_prev_id, 1] if _prev_id else ["5", 0]
        refchain_nodes[node_id] = {
            "class_type": "ReferenceChainConditioning",
            "inputs": {
                "conditioning":           cond_input,
                "neg_conditioning":       neg_input,
                "vae":                    ["3", 0],
                "upscale_method":         "lanczos",
                "scale_megapixels":       megapixels,
                "images":                 json.dumps(fnames),
                "image_input_node_count": 0,
            },
            "_meta": {"title": f"ReferenceChain — {char_name}"},
        }
        _prev_id = node_id

    # Last node in the chain feeds CFGGuider.
    _cfg_pos = [_prev_id, 0] if _prev_id else ["4", 0]
    _cfg_neg = [_prev_id, 1] if _prev_id else ["5", 0]

    return {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_name},
            "_meta": {"title": "UNET Loader (GGUF)"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
            "_meta": {"title": "CLIP Loader"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "VAE Loader"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": enhanced_prompt},
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        # FLUX.2 Klein is a distilled model — zero out the negative conditioning.
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
            "_meta": {"title": "Zero Out (Negative)"},
        },
        # Per-character ReferenceChainConditioning nodes (dynamically built above).
        **refchain_nodes,
        "31": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "_meta": {"title": "Empty Flux2 Latent Image"},
        },
        "32": {
            "class_type": "Flux2Scheduler",
            "inputs": {"steps": steps, "width": width, "height": height},
            "_meta": {"title": "Flux2 Scheduler"},
        },
        "33": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
            "_meta": {"title": "KSampler Select"},
        },
        "34": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
            "_meta": {"title": "Random Noise"},
        },
        # CFGGuider (cfg=1.0) — correct pattern for distilled FLUX.2 Klein.
        # Positive/negative wired from the last ReferenceChainConditioning in the chain.
        "35": {
            "class_type": "CFGGuider",
            "inputs": {
                "model":    ["1", 0],
                "positive": _cfg_pos,
                "negative": _cfg_neg,
                "cfg":      1.0,
            },
            "_meta": {"title": "CFG Guider"},
        },
        "36": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise":        ["34", 0],
                "guider":       ["35", 0],
                "sampler":      ["33", 0],
                "sigmas":       ["32", 0],
                "latent_image": ["31", 0],
            },
            "_meta": {"title": "Sampler Custom Advanced"},
        },
        "37": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
            "_meta": {"title": "VAE Decode"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["37", 0], "filename_prefix": "ariabot_"},
            "_meta": {"title": "Save Image"},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRUE multi-character photo-referencing workflow (no SAM3, no custom nodes)
# ─────────────────────────────────────────────────────────────────────────────

def build_multiref_workflow(
    scene_prompt: str,
    subject_filenames: Dict[str, List[str]],
    subject_appearances: Dict[str, str],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 6,
    width: int = 1280,
    height: int = 720,
    contact_pose: bool = False,
) -> Dict[str, dict]:
    """TRUE multi-character photo-referencing using ClownRegionalConditioning_AB for spatial masking.

    Architecture overview
    ─────────────────────
    For EACH character:
        CLIPTextEncode (scene + char-specific appearance text)
        → ReferenceLatent chain (one node per reference photo, positive + negative)

    For 2 characters → ClownRegionalConditioning_AB:
        Char 0 conditioning locked to left half  (SolidMask + MaskComposite)
        Char 1 conditioning locked to right half
        Output: single CONDITIONING fed to CFGGuider

    For 1 character → plain ReferenceLatent chain, no spatial split.
    For 3+ characters → ConditioningCombine fallback (no spatial masks).

    Spatial masks are built entirely with ComfyUI nodes (SolidMask → MaskComposite),
    no file uploads needed.

    Node-ID scheme
    ──────────────
    Fixed:      "1"–"3"    model loaders
                "5"        shared ConditioningZeroOut (negative anchor)
    Per char s: "TC{s}"    CLIPTextEncode  (scene + char appearance)
                "L{s}_{p}" LoadImage
                "SC{s}_{p}" ImageScaleToTotalPixels
                "E{s}_{p}"  VAEEncode
                "RP{s}_{p}" ReferenceLatent positive chain
                "RN{s}_{p}" ReferenceLatent negative chain
    2-char masks:"MB0"      SolidMask black base (width × height)
                "MB1"      SolidMask white left strip (width//2 × height)
                "MB2"      SolidMask white right strip ((width-width//2) × height)
                "ML"       MaskComposite → left-half mask
                "MR"       MaskComposite → right-half mask
                "RCA"      ClownRegionalConditioning_AB (positive)
                "RCN"      ConditioningCombine (negative)
    3+char:     "CP{k}"    ConditioningCombine positive merge
                "CN{k}"    ConditioningCombine negative merge
    Output:     "31"–"37", "9"
    """
    subjects: List[Tuple[str, List[str]]] = [
        (name, fnames) for name, fnames in subject_filenames.items() if fnames
    ]

    megapixels = round((width * height) / 1_000_000, 4)
    half_w = width // 2

    # ── Shared model loader nodes + global scene conditioning ────────────────
    workflow: Dict[str, Any] = {
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_name},
            "_meta": {"title": "UNET Loader (GGUF)"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "flux2"},
            "_meta": {"title": "CLIP Loader"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "VAE Loader"},
        },
        # Global scene conditioning — no mask, covers the whole canvas.
        # Guides the background and ensures a cohesive image regardless of
        # how many characters are present.
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": scene_prompt + _ANATOMY_SUFFIX},
            "_meta": {"title": "Global scene prompt"},
        },
        # Zero-out negative anchored to the global scene text.
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
            "_meta": {"title": "Zero Out (Negative anchor)"},
        },
    }

    # ── Per-character conditioning chains ────────────────────────────────────
    _SIDE_LABELS = ["Left character", "Right character", "Center character", "Background character"]
    char_final_pos: List[str] = []
    char_final_neg: List[str] = []

    for s, (char_name, fnames) in enumerate(subjects):
        # Character-specific text: scene context + spatial label + appearance
        app = subject_appearances.get(char_name, "").strip()
        if len(subjects) > 1 and s < len(_SIDE_LABELS):
            label = f"{_SIDE_LABELS[s]} — {char_name}"
        else:
            label = char_name
        char_text = f"{scene_prompt}. {label}: {app}" if app else f"{scene_prompt}. {label}"
        char_text += _ANATOMY_SUFFIX

        tc_id = f"TC{s}"
        workflow[tc_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": char_text},
            "_meta": {"title": f"Text — {char_name}"},
        }

        # ReferenceLatent chains: positive starts from char text, negative from zero-out
        for p, img_name in enumerate(fnames):
            load_id  = f"L{s}_{p}"
            scale_id = f"SC{s}_{p}"
            enc_id   = f"E{s}_{p}"
            rpos_id  = f"RP{s}_{p}"
            rneg_id  = f"RN{s}_{p}"
            prev_pos: List = [tc_id, 0] if p == 0 else [f"RP{s}_{p-1}", 0]
            prev_neg: List = ["5", 0]   if p == 0 else [f"RN{s}_{p-1}", 0]

            workflow[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name, "upload": "image"},
                "_meta": {"title": f"Load — {char_name} ref #{p + 1}"},
            }
            workflow[scale_id] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": [load_id, 0], "upscale_method": "lanczos",
                           "megapixels": megapixels, "resolution_steps": 64},
                "_meta": {"title": f"Scale — {char_name} ref #{p + 1}"},
            }
            workflow[enc_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
                "_meta": {"title": f"VAE Encode — {char_name} ref #{p + 1}"},
            }
            workflow[rpos_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_pos, "latent": [enc_id, 0]},
                "_meta": {"title": f"ReferenceLatent+ — {char_name} #{p + 1}"},
            }
            workflow[rneg_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_neg, "latent": [enc_id, 0]},
                "_meta": {"title": f"ReferenceLatent- — {char_name} #{p + 1}"},
            }

        last_p = len(fnames) - 1
        char_final_pos.append(f"RP{s}_{last_p}")
        char_final_neg.append(f"RN{s}_{last_p}")

    # ── Spatial merge ────────────────────────────────────────────────────────
    # contact_pose=True skips spatial masks entirely so characters can
    # physically interact — falls through to plain ConditioningCombine below.
    if len(subjects) == 2 and not contact_pose:
        # SolidMask nodes build left/right half masks entirely in-graph —
        # no file uploads required.
        #
        #   MB0 (black, full size)
        #   MB1 (white, left half)  ──MaskComposite(add, x=0)──▶  ML (left mask)
        #   MB2 (white, right half) ──MaskComposite(add, x=right_x)─▶ MR (right mask)
        #
        # 48-pixel overlap at the centre boundary so both conditionings are
        # active in that zone.  Used by both hard and soft spatial paths.
        overlap = 48
        left_w  = half_w + overlap
        right_x = max(0, half_w - overlap)
        right_w = width - right_x

        workflow["MB0"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 0.0, "width": width, "height": height},
            "_meta": {"title": "Mask base (black)"},
        }
        workflow["MB1"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 1.0, "width": left_w, "height": height},
            "_meta": {"title": "Left white strip (+ overlap)"},
        }
        workflow["MB2"] = {
            "class_type": "SolidMask",
            "inputs": {"value": 1.0, "width": right_w, "height": height},
            "_meta": {"title": "Right white strip (+ overlap)"},
        }
        workflow["ML"] = {
            "class_type": "MaskComposite",
            "inputs": {"destination": ["MB0", 0], "source": ["MB1", 0],
                       "x": 0, "y": 0, "operation": "add"},
            "_meta": {"title": "Left-half mask (with centre overlap)"},
        }
        workflow["MR"] = {
            "class_type": "MaskComposite",
            "inputs": {"destination": ["MB0", 0], "source": ["MB2", 0],
                       "x": right_x, "y": 0, "operation": "add"},
            "_meta": {"title": "Right-half mask (with centre overlap)"},
        }

        if not contact_pose:
            # ── Hard spatial masks (default: side-by-side scenes) ─────────────
            # "mask bounds" mode: each character's conditioning is clipped to
            # their half.  Gives best feature accuracy for non-contact poses.
            #
            # ConditioningSetAreaStrength (strength=2.0) sits between the
            # ReferenceLatent chain and the mask so the photo reference gets
            # 2× attention weight vs text, ensuring consistent character
            # appearance regardless of seed.
            workflow["AS0"] = {
                "class_type": "ConditioningSetAreaStrength",
                "inputs": {"conditioning": [char_final_pos[0], 0], "strength": 2.0},
                "_meta": {"title": "Boost ref strength — char 0"},
            }
            workflow["AS1"] = {
                "class_type": "ConditioningSetAreaStrength",
                "inputs": {"conditioning": [char_final_pos[1], 0], "strength": 2.0},
                "_meta": {"title": "Boost ref strength — char 1"},
            }
            workflow["SM0"] = {
                "class_type": "ConditioningSetMask",
                "inputs": {
                    "conditioning":  ["AS0", 0],
                    "mask":          ["ML", 0],
                    "strength":      1.0,
                    "set_cond_area": "mask bounds",
                },
                "_meta": {"title": "Hard mask char 0 → left half"},
            }
            workflow["SM1"] = {
                "class_type": "ConditioningSetMask",
                "inputs": {
                    "conditioning":  ["AS1", 0],
                    "mask":          ["MR", 0],
                    "strength":      1.0,
                    "set_cond_area": "mask bounds",
                },
                "_meta": {"title": "Hard mask char 1 → right half"},
            }
            workflow["RCA0"] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["SM0", 0], "conditioning_2": ["SM1", 0]},
                "_meta": {"title": "Combine hard-masked conditionings"},
            }
            # Strip character names from node "4" so the unmasked global
            # conditioning only guides background/atmosphere, not characters.
            # Without this, the model regenerates both characters globally on
            # top of the already-masked per-character conditionings → ghosts.
            _bg = scene_prompt
            for _n in subject_appearances.keys():
                _bg = re.sub(rf'\b{re.escape(_n)}\b', '', _bg, flags=re.IGNORECASE)
            _bg = re.sub(r'\b(and|,)\b\s*', ' ', _bg)   # clean up orphaned "and" / commas
            _bg = ' '.join(_bg.split())                   # collapse whitespace
            workflow["4"]["inputs"]["text"] = _bg + _ANATOMY_SUFFIX

            # Scene node covers the full canvas with no mask.  At strength 1.0
            # it spawns ghost characters; throttled to 0.3 it only guides
            # background / atmosphere.
            workflow["SC4"] = {
                "class_type": "ConditioningSetAreaStrength",
                "inputs": {"conditioning": ["4", 0], "strength": 0.3},
                "_meta": {"title": "Scene at 0.3 — background only, no ghost chars"},
            }
            workflow["RCA"] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["RCA0", 0], "conditioning_2": ["SC4", 0]},
                "_meta": {"title": "Add scene (reduced) to hard-masked chars"},
            }
            workflow["RCN"] = {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [char_final_neg[0], 0],
                    "conditioning_2": [char_final_neg[1], 0],
                },
                "_meta": {"title": "Negative merge"},
            }
            final_pos: List = ["RCA", 0]
            final_neg: List = ["RCN", 0]
            print("[MultiRef] 2-char hard spatial masks + global scene (side-by-side mode)")

        else:
            # ── Soft spatial masks (contact_pose: hugging / touching) ─────────
            # "default" area mode: conditioning is NOT hard-clipped to the mask;
            # instead the mask acts as a weight so each character's conditioning
            # bleeds softly across the boundary.  Strength 0.65 keeps enough
            # spatial bias to prevent feature-blending while allowing arms /
            # bodies to cross the half-way line naturally.
            workflow["SM0"] = {
                "class_type": "ConditioningSetMask",
                "inputs": {
                    "conditioning":  [char_final_pos[0], 0],
                    "mask":          ["ML", 0],
                    "strength":      0.65,
                    "set_cond_area": "default",
                },
                "_meta": {"title": "Soft mask char 0 → left (contact mode)"},
            }
            workflow["SM1"] = {
                "class_type": "ConditioningSetMask",
                "inputs": {
                    "conditioning":  [char_final_pos[1], 0],
                    "mask":          ["MR", 0],
                    "strength":      0.65,
                    "set_cond_area": "default",
                },
                "_meta": {"title": "Soft mask char 1 → right (contact mode)"},
            }
            workflow["RCA0"] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["SM0", 0], "conditioning_2": ["SM1", 0]},
                "_meta": {"title": "Combine soft-masked conditionings"},
            }
            workflow["RCA"] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": ["RCA0", 0], "conditioning_2": ["4", 0]},
                "_meta": {"title": "Add global scene (contact mode)"},
            }
            workflow["RCN"] = {
                "class_type": "ConditioningCombine",
                "inputs": {
                    "conditioning_1": [char_final_neg[0], 0],
                    "conditioning_2": [char_final_neg[1], 0],
                },
                "_meta": {"title": "Negative merge"},
            }
            final_pos: List = ["RCA", 0]
            final_neg: List = ["RCN", 0]
            print("[MultiRef] 2-char soft spatial masks + global scene (contact-pose mode)")

    else:
        # 1 char or 3+ chars: plain ConditioningCombine (no spatial split)
        def _merge(id_list: List[str], prefix: str) -> List:
            if not id_list:
                return ["5", 0]
            if len(id_list) == 1:
                return [id_list[0], 0]
            cur = id_list[0]
            for k in range(1, len(id_list)):
                mid = f"{prefix}{k - 1}"
                workflow[mid] = {
                    "class_type": "ConditioningCombine",
                    "inputs": {"conditioning_1": [cur, 0], "conditioning_2": [id_list[k], 0]},
                    "_meta": {"title": f"Merge — step {k}"},
                }
                cur = mid
            return [cur, 0]

        print(f"[MultiRef] {len(subjects)}-char plain combine (no spatial split)")
        final_pos = _merge(char_final_pos, "CP")
        final_neg = _merge(char_final_neg, "CN")

    # Standard FLUX.2 Klein advanced sampler pipeline
    workflow["31"] = {
        "class_type": "EmptyFlux2LatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
        "_meta": {"title": "Empty Flux2 Latent Image"},
    }
    workflow["32"] = {
        "class_type": "Flux2Scheduler",
        "inputs": {"steps": steps, "width": width, "height": height},
        "_meta": {"title": "Flux2 Scheduler"},
    }
    workflow["33"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler"},
        "_meta": {"title": "KSampler Select"},
    }
    workflow["34"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
        "_meta": {"title": "Random Noise"},
    }
    workflow["35"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["1", 0],
            "positive": final_pos,
            "negative": final_neg,
            "cfg": 1.0,
        },
        "_meta": {"title": "CFG Guider (cfg=1.0)"},
    }
    workflow["36"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise":        ["34", 0],
            "guider":       ["35", 0],
            "sampler":      ["33", 0],
            "sigmas":       ["32", 0],
            "latent_image": ["31", 0],
        },
        "_meta": {"title": "Sampler Custom Advanced"},
    }
    workflow["37"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
        "_meta": {"title": "VAE Decode"},
    }
    workflow["9"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["37", 0], "filename_prefix": "ariabot_"},
        "_meta": {"title": "Save Image"},
    }

    n_chars = len(subjects)
    n_photos = sum(len(f) for _, f in subjects)
    print(
        f"[MultiRef] Built multiref workflow — {n_chars} character(s), "
        f"{n_photos} reference photo(s) total, output={width}x{height}, "
        f"appearance_text={[c for c, _ in subjects if subject_appearances.get(c)]}"
    )
    return workflow


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS INPAINT HELPERS (no SAM, no ControlNet)
# ─────────────────────────────────────────────────────────────────────────────

def build_layout_workflow(
    scene_prompt: str,
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 4,
    width: int = 512,
    height: int = 768,
) -> Dict[str, dict]:
    """Pass 1 — generate a layout image from scene text only.

    No character refs, no spatial masks.  The model freely picks a pose
    (hugging, standing, etc.) driven purely by the scene description.
    The resulting image is fed into build_inpaint_char_workflow() twice —
    once per character — to paint in the correct appearance.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": unet_name},
              "_meta": {"title": "UNET Loader (GGUF)"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip_name, "type": "flux2"},
              "_meta": {"title": "CLIP Loader"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae_name},
              "_meta": {"title": "VAE Loader"}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": scene_prompt + _ANATOMY_SUFFIX},
              "_meta": {"title": "Scene prompt"}},
        "5": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["4", 0]},
              "_meta": {"title": "Zero Out (negative)"}},
        "31": {"class_type": "EmptyFlux2LatentImage",
               "inputs": {"width": width, "height": height, "batch_size": 1},
               "_meta": {"title": "Empty Latent"}},
        "32": {"class_type": "Flux2Scheduler",
               "inputs": {"steps": steps, "width": width, "height": height},
               "_meta": {"title": "Flux2 Scheduler"}},
        "33": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"},
               "_meta": {"title": "Sampler Select"}},
        "34": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": seed},
               "_meta": {"title": "Random Noise"}},
        "35": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["4", 0],
                          "negative": ["5", 0], "cfg": 1.0},
               "_meta": {"title": "CFG Guider"}},
        "36": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["34", 0], "guider": ["35", 0],
                          "sampler": ["33", 0], "sigmas": ["32", 0],
                          "latent_image": ["31", 0]},
               "_meta": {"title": "Sampler"}},
        "37": {"class_type": "VAEDecode",
               "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
               "_meta": {"title": "VAE Decode"}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"images": ["37", 0], "filename_prefix": "ariabot_layout_"},
               "_meta": {"title": "Save Layout"}},
    }


def build_inpaint_char_workflow(
    layout_image_filename: str,
    char_name: str,
    char_filenames: List[str],
    char_appearance: str,
    scene_prompt: str,
    side: str,           # "left" or "right"
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 6,
    width: int = 512,
    height: int = 768,
    overlap: int = 96,   # px each side overlaps centre — wider for contact poses
) -> Dict[str, dict]:
    """Pass 2 — inpaint one character into their half of the layout image.

    Workflow:
        LoadImage(layout) → VAEEncodeForInpaint(+mask) → SetLatentNoiseMask
        Character CLIPTextEncode → ReferenceLatent chain
        CFGGuider → SamplerCustomAdvanced → VAEDecode → SaveImage

    The mask for each character's half includes an `overlap` px zone at the
    centre so the inpainting blends naturally where arms / bodies cross.
    """
    half_w = width // 2
    if side == "left":
        # white strip from x=0 to x=half_w+overlap
        mask_w   = half_w + overlap
        mask_x   = 0
    else:
        # white strip from x=half_w-overlap to x=width
        mask_w   = (width - half_w) + overlap
        mask_x   = max(0, half_w - overlap)

    # ── Character text conditioning ─────────────────────────────────────────
    side_label = "Left character" if side == "left" else "Right character"
    char_text = f"{scene_prompt}. {side_label} — {char_name}: {char_appearance}" \
                if char_appearance.strip() else f"{scene_prompt}. {side_label} — {char_name}"
    char_text += _ANATOMY_SUFFIX

    wf: Dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": unet_name},
              "_meta": {"title": "UNET Loader (GGUF)"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip_name, "type": "flux2"},
              "_meta": {"title": "CLIP Loader"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae_name},
              "_meta": {"title": "VAE Loader"}},
        # Load layout image from ComfyUI input dir
        "LI": {"class_type": "LoadImage",
               "inputs": {"image": layout_image_filename, "upload": "image"},
               "_meta": {"title": f"Layout image (pass 1 output)"}},
        # Build the inpaint mask in-graph
        "MB0": {"class_type": "SolidMask",
                "inputs": {"value": 0.0, "width": width, "height": height},
                "_meta": {"title": "Mask base (black)"}},
        "MB1": {"class_type": "SolidMask",
                "inputs": {"value": 1.0, "width": mask_w, "height": height},
                "_meta": {"title": f"{side} white strip"}},
        "ML":  {"class_type": "MaskComposite",
                "inputs": {"destination": ["MB0", 0], "source": ["MB1", 0],
                           "x": mask_x, "y": 0, "operation": "add"},
                "_meta": {"title": f"{side} inpaint mask"}},
        # VAE-encode layout + mask → inpaint latent
        "VI": {"class_type": "VAEEncodeForInpaint",
               "inputs": {"pixels": ["LI", 0], "vae": ["3", 0],
                          "mask": ["ML", 0], "grow_mask_by": 6},
               "_meta": {"title": "VAE Encode For Inpaint"}},
        # Mark masked region for denoising
        "NM": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["VI", 0], "mask": ["ML", 0]},
               "_meta": {"title": "Set Noise Mask"}},
        # Character text conditioning
        "TC": {"class_type": "CLIPTextEncode",
               "inputs": {"clip": ["2", 0], "text": char_text},
               "_meta": {"title": f"Text — {char_name}"}},
        "5":  {"class_type": "ConditioningZeroOut",
               "inputs": {"conditioning": ["TC", 0]},
               "_meta": {"title": "Zero Out (negative)"}},
    }

    # ── ReferenceLatent chain (one node per reference photo) ─────────────────
    prev_pos: List = ["TC", 0]
    prev_neg: List = ["5",  0]
    for p, img_name in enumerate(char_filenames):
        enc_id = f"IE{p}"
        rl_id  = f"RL{p}"
        rn_id  = f"RN{p}"
        wf[enc_id] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": [f"LR{p}", 0], "vae": ["3", 0]},  # placeholder
            "_meta": {"title": f"VAE Encode ref {p}"},
        }
        # LoadImage for each reference photo
        wf[f"LR{p}"] = {
            "class_type": "LoadImage",
            "inputs": {"image": img_name, "upload": "image"},
            "_meta": {"title": f"Ref photo {p} — {char_name}"},
        }
        wf[rl_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_pos, "latent": [enc_id, 0]},
            "_meta": {"title": f"ReferenceLatent + {p}"},
        }
        wf[rn_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_neg, "latent": [enc_id, 0]},
            "_meta": {"title": f"ReferenceLatent neg {p}"},
        }
        prev_pos = [rl_id, 0]
        prev_neg = [rn_id, 0]

    # ── Sampler ──────────────────────────────────────────────────────────────
    wf["32"] = {"class_type": "Flux2Scheduler",
                "inputs": {"steps": steps, "width": width, "height": height},
                "_meta": {"title": "Flux2 Scheduler"}}
    wf["33"] = {"class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
                "_meta": {"title": "Sampler Select"}}
    wf["34"] = {"class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
                "_meta": {"title": "Random Noise"}}
    wf["35"] = {"class_type": "CFGGuider",
                "inputs": {"model": ["1", 0], "positive": prev_pos,
                           "negative": prev_neg, "cfg": 1.0},
                "_meta": {"title": "CFG Guider"}}
    wf["36"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["34", 0], "guider": ["35", 0],
                           "sampler": ["33", 0], "sigmas": ["32", 0],
                           "latent_image": ["NM", 0]},
                "_meta": {"title": "Sampler"}}
    wf["37"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
                "_meta": {"title": "VAE Decode"}}
    wf["9"]  = {"class_type": "SaveImage",
                "inputs": {"images": ["37", 0],
                           "filename_prefix": f"ariabot_inpaint_{side}_"},
                "_meta": {"title": f"Save — {char_name} inpainted"}}

    print(f"[Inpaint] Built inpaint workflow — {char_name} ({side}), "
          f"{len(char_filenames)} ref(s), {steps} steps, overlap={overlap}px")
    return wf


# ─────────────────────────────────────────────────────────────────────────────
# TWO-PASS img2img: side-by-side → hugging pose transfer
# ─────────────────────────────────────────────────────────────────────────────

def build_img2img_workflow(
    input_image_filename: str,
    scene_prompt: str,
    subject_filenames: Dict[str, List[str]],
    subject_appearances: Dict[str, str],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 4,
    width: int = 512,
    height: int = 768,
    layout_ref_weight: float = 1.0,
) -> Dict[str, dict]:
    """Pass 2 — reference-guided pose-transfer on top of a proven side-by-side.

    FLUX.2 Klein is a distilled 4-step model that must always run its FULL
    sigma schedule (σ_max → σ_0).  Standard img2img via SplitSigmasDenoise
    does NOT work — the model produces pure noise when started mid-schedule.

    Instead, this workflow VAE-encodes the Pass-1 image and feeds it as an
    additional ReferenceLatent node for EVERY character's conditioning chain.
    The sampler runs the normal 4-step schedule from scratch, but the model's
    self-attention is anchored to the Pass-1 composition (characters' positions,
    lighting, background) while the photo ReferenceLatents preserve appearance.

    The hugging scene prompt + no spatial masks let the model freely redraw the
    characters closer together.

    Node layout:
        LoadImage(pass1) → Scale → VAEEncode → layout_latent
        Per char: CLIPTextEncode → ReferenceLatent(photo) → ReferenceLatent(layout)
        ConditioningCombine (both chars) → CFGGuider
        EmptyFlux2LatentImage → Flux2Scheduler → SamplerCustomAdvanced
        VAEDecode → SaveImage
    """
    megapixels = round((width * height) / 1_000_000, 4)

    subjects: List[Tuple[str, List[str]]] = [
        (name, fnames) for name, fnames in subject_filenames.items() if fnames
    ]

    wf: Dict[str, Any] = {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": unet_name},
              "_meta": {"title": "UNET Loader (GGUF)"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": clip_name, "type": "flux2"},
              "_meta": {"title": "CLIP Loader"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae_name},
              "_meta": {"title": "VAE Loader"}},
        # Load & encode Pass-1 side-by-side as structural reference
        "LI": {"class_type": "LoadImage",
               "inputs": {"image": input_image_filename, "upload": "image"},
               "_meta": {"title": "Pass-1 image (structural ref)"}},
        "SI": {"class_type": "ImageScaleToTotalPixels",
               "inputs": {"image": ["LI", 0], "upscale_method": "lanczos",
                          "megapixels": megapixels, "resolution_steps": 64},
               "_meta": {"title": "Scale Pass-1 to target res"}},
        "VI": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["SI", 0], "vae": ["3", 0]},
               "_meta": {"title": "VAE Encode Pass-1 → layout latent"}},
        # Global scene + negative
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": scene_prompt + _ANATOMY_SUFFIX},
              "_meta": {"title": "Global scene prompt"}},
        "5": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["4", 0]},
              "_meta": {"title": "Zero Out (negative anchor)"}},
    }

    # ── Per-character chains: photo refs → layout ref ─────────────────────────
    char_final_pos: List[Any] = []
    char_final_neg: List[Any] = []

    for s, (char_name, fnames) in enumerate(subjects):
        app = subject_appearances.get(char_name, "").strip()
        char_text = (f"{scene_prompt}. {char_name}: {app}" if app
                     else f"{scene_prompt}. {char_name}")
        char_text += _ANATOMY_SUFFIX

        tc_id = f"TC{s}"
        wf[tc_id] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": char_text},
            "_meta": {"title": f"Text — {char_name}"},
        }

        # Photo reference latents (appearance)
        prev_pos: List = [tc_id, 0]
        prev_neg: List = ["5",  0]
        for p, img_name in enumerate(fnames):
            load_id  = f"L{s}_{p}"
            scale_id = f"SC{s}_{p}"
            enc_id   = f"E{s}_{p}"
            rpos_id  = f"RP{s}_{p}"
            rneg_id  = f"RN{s}_{p}"

            wf[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": img_name, "upload": "image"},
                "_meta": {"title": f"Photo ref — {char_name} #{p + 1}"},
            }
            wf[scale_id] = {
                "class_type": "ImageScaleToTotalPixels",
                "inputs": {"image": [load_id, 0], "upscale_method": "lanczos",
                           "megapixels": megapixels, "resolution_steps": 64},
                "_meta": {"title": f"Scale photo — {char_name} #{p + 1}"},
            }
            wf[enc_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
                "_meta": {"title": f"Encode photo — {char_name} #{p + 1}"},
            }
            wf[rpos_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_pos, "latent": [enc_id, 0]},
                "_meta": {"title": f"RefLatent+ photo — {char_name} #{p + 1}"},
            }
            wf[rneg_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": prev_neg, "latent": [enc_id, 0]},
                "_meta": {"title": f"RefLatent- photo — {char_name} #{p + 1}"},
            }
            prev_pos = [rpos_id, 0]
            prev_neg = [rneg_id, 0]

        # Layout reference latent (structural hint from Pass 1)
        lay_pos_id = f"LP{s}"
        lay_neg_id = f"LN{s}"
        wf[lay_pos_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_pos, "latent": ["VI", 0]},
            "_meta": {"title": f"RefLatent+ layout — {char_name}"},
        }
        wf[lay_neg_id] = {
            "class_type": "ReferenceLatent",
            "inputs": {"conditioning": prev_neg, "latent": ["VI", 0]},
            "_meta": {"title": f"RefLatent- layout — {char_name}"},
        }

        # Boost reference weight so photo appearance dominates over text drift
        ias_id = f"IAS{s}"
        wf[ias_id] = {
            "class_type": "ConditioningSetAreaStrength",
            "inputs": {"conditioning": [lay_pos_id, 0], "strength": 2.0},
            "_meta": {"title": f"Boost ref strength — {char_name} (img2img)"},
        }
        char_final_pos.append([ias_id, 0])
        char_final_neg.append([lay_neg_id, 0])

    # ── Merge all character conditionings (plain combine — no spatial masks) ──
    def _combine(refs: List[Any], prefix: str) -> Any:
        if len(refs) == 1:
            return refs[0]
        cur = refs[0]
        for k in range(1, len(refs)):
            mid = f"{prefix}{k}"
            wf[mid] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": cur, "conditioning_2": refs[k]},
                "_meta": {"title": f"Combine {prefix} {k}"},
            }
            cur = [mid, 0]
        return cur

    final_pos = _combine(char_final_pos, "CP")
    final_neg = _combine(char_final_neg, "CN")

    # ── Sampler: standard 4-step full-schedule run ────────────────────────────
    wf["31"] = {"class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "_meta": {"title": "Empty Latent"}}
    wf["32"] = {"class_type": "Flux2Scheduler",
                "inputs": {"steps": steps, "width": width, "height": height},
                "_meta": {"title": "Flux2 Scheduler"}}
    wf["33"] = {"class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler"},
                "_meta": {"title": "KSampler Select"}}
    wf["34"] = {"class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
                "_meta": {"title": "Random Noise"}}
    wf["35"] = {"class_type": "CFGGuider",
                "inputs": {"model": ["1", 0], "positive": final_pos,
                           "negative": final_neg, "cfg": 1.0},
                "_meta": {"title": "CFG Guider"}}
    wf["36"] = {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise":        ["34", 0],
                           "guider":       ["35", 0],
                           "sampler":      ["33", 0],
                           "sigmas":       ["32", 0],
                           "latent_image": ["31", 0]},
                "_meta": {"title": "Sampler"}}
    wf["37"] = {"class_type": "VAEDecode",
                "inputs": {"samples": ["36", 0], "vae": ["3", 0]},
                "_meta": {"title": "VAE Decode"}}
    wf["9"]  = {"class_type": "SaveImage",
                "inputs": {"images": ["37", 0],
                           "filename_prefix": "ariabot_i2i_"},
                "_meta": {"title": "Save result"}}

    n_chars = len(subjects)
    n_photos = sum(len(f) for _, f in subjects)
    print(f"[Img2Img] Built ref-guided workflow — {n_chars} char(s), "
          f"{n_photos} photo ref(s) + Pass-1 layout ref, steps={steps}, {width}×{height}")
    return wf
