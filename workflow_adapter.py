"""
ComfyUI workflow adapters for AriaBot.

Primary: build_refchain_workflow() — programmatic FLUX.2 Klein workflow using
  ReferenceChainConditioning (ComfyUI-ReferenceChain). One node accepts all
  reference images at once as a JSON list, handles scaling+VAE-encoding internally,
  and chains them as reference latents. Requires one node pack; no JSON file needed.

Legacy (kept for reference): AIO per-segment inpainting workflow expander.
  Converts the subgraph-format "Flux.2 AIO Pro Simple v3.1" GUI workflow JSON
  into the flat API format. Requires 6+ custom node packs; no longer used as
  the primary path.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Anatomy quality suffix — must match _ANATOMY_SUFFIX in comfyui_ai.py.
# Duplicated here to avoid a circular import (comfyui_ai imports workflow_adapter).
_ANATOMY_SUFFIX = (
    ", perfect anatomy, correct arms, well-drawn hands, "
    "five fingers, proper limbs, symmetrical body"
)

_AIO_PATH = os.path.join(os.path.dirname(__file__), "workflows", "Flux_2_AIO_Pro_Simple_v3_1.json")

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


def build_refchain_workflow(
    prompt: str,
    uploaded_filenames: List[str],
    unet_name: str,
    vae_name: str,
    clip_name: str,
    seed: int,
    steps: int = 6,
    width: int = 1280,
    height: int = 720,
) -> Dict[str, dict]:
    """Build a ComfyUI API workflow using ReferenceChainConditioning.

    This is the primary multi-reference workflow for FLUX.2 Klein.  The
    ReferenceChainConditioning node (from ComfyUI-ReferenceChain,
    https://github.com/remingtonspaz/ComfyUI-ReferenceChain) accepts all
    reference images at once as a JSON list of ComfyUI-uploaded filenames,
    handles scaling + VAE-encoding internally, and chains them as reference
    latents into a single conditioning output.

    Advantages over the AIO inpainting approach:
      • One node pack instead of six
      • No GUI JSON file — workflow is pure Python
      • No inpainting loop or mask machinery required
      • Both positive AND negative conditioning are updated together (matching
        how FLUX.2 Klein expects reference latents on both streams)

    Node inputs for ReferenceChainConditioning (from nodes.py):
      conditioning         — positive CONDITIONING from CLIPTextEncode
      neg_conditioning     — negative CONDITIONING (zero-out for distilled FLUX)
      vae                  — VAE model
      upscale_method       — "lanczos" (best quality)
      scale_megapixels     — matched to output resolution (width*height/1e6)
      images               — JSON string list of ComfyUI uploaded filenames
      image_input_node_count — 0 (no external IMAGE node inputs)

    Outputs (slot indices):
      0: conditioning      — positive conditioning with all reference latents
      1: neg_conditioning  — negative conditioning with all reference latents
      2: first_image_scaled — not connected

    Args:
        prompt:             Raw text prompt (anatomy suffix appended internally).
        uploaded_filenames: ComfyUI server filenames of all uploaded reference images,
                            in order. All characters' photos in a flat list.
        unet_name:          GGUF model filename for UnetLoaderGGUF.
        vae_name:           VAE filename for VAELoader.
        clip_name:          CLIP filename for CLIPLoader.
        seed:               Random noise seed.
        steps:              Sampler step count.
        width / height:     Output resolution.

    Returns:
        ComfyUI API-format workflow dict, ready for the /prompt endpoint.
    """
    enhanced_prompt = prompt + _ANATOMY_SUFFIX
    megapixels = round((width * height) / 1_000_000, 4)

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
        # ReferenceChainConditioning: all reference images → reference latents on both
        # conditioning streams.  The `images` input is a JSON list of ComfyUI filenames.
        "6": {
            "class_type": "ReferenceChainConditioning",
            "inputs": {
                "conditioning": ["4", 0],
                "neg_conditioning": ["5", 0],
                "vae": ["3", 0],
                "upscale_method": "lanczos",
                "scale_megapixels": megapixels,
                "images": json.dumps(uploaded_filenames),
                "image_input_node_count": 0,
            },
            "_meta": {"title": "Reference Chain Conditioning"},
        },
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
        # CFGGuider with cfg=1.0 (effectively no CFG amplification) and a zeroed-out
        # negative — this is the proven local pattern for distilled FLUX.2 Klein.
        # ReferenceChainConditioning outputs both pos (slot 0) and neg (slot 1)
        # conditioning streams; wiring both to CFGGuider is more correct than
        # BasicGuider+FluxGuidance (which discards the neg_conditioning output).
        # CFGGuider: slot 0 → positive conditioning (refchain output 0)
        #            slot 1 → negative conditioning (refchain output 1)
        "35": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "cfg": 1.0,
            },
            "_meta": {"title": "CFG Guider"},
        },
        "36": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["34", 0],
                "guider": ["35", 0],
                "sampler": ["33", 0],
                "sigmas": ["32", 0],
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
