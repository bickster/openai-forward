# FLUX.1 Kontext Image Editing Implementation Plan

## Overview
Implement forward call support for FLUX.1 Kontext image editing API to handle `/images/edits` requests when `IMAGE_EDIT_PLATFORM=flux1_kontext`.

## Implementation Steps

### 1. Create FluxKontext Class
- Add new `FluxKontext` class in `/openai_forward/flux/bfl_api.py`
- Extend `FluxBase` with Kontext-specific endpoints and logic
- Handle image editing requests with base64 input images and edit prompts

### 2. Update Request Routing Logic
- Modify `base.py` to route `/images/edits` requests to FLUX.1 Kontext when configured
- Add proper match case for `ImageEditPlatform.flux1_kontext` 
- Handle ContentModerationError responses consistently

### 3. Request/Response Translation
- Transform OpenAI image edit request format to FLUX.1 Kontext API format
- Convert `image` (base64) and `prompt` from OpenAI format to Kontext format
- Handle optional parameters like `size`, `n`, `response_format`
- Return OpenAI-compatible response format

### 4. Environment Configuration
- Support `BFL_API_KEY` environment variable (already exists)
- Document `IMAGE_EDIT_PLATFORM=flux1_kontext` configuration option

## Key Technical Details
- FLUX.1 Kontext uses `/flux-kontext-pro` endpoint
- Requires `x-key` header authentication
- Input image must be base64 encoded (up to 20MB/20MP)
- Uses polling mechanism similar to existing FLUX.1.1 implementation
- Returns result via signed URL (valid 10 minutes)

## API Documentation Reference
From https://docs.bfl.ai/kontext/kontext_image_editing:

### Endpoint
- `/flux-kontext-pro`

### Authentication
- Requires API key in `x-key` header
- API key obtained through Black Forest Labs dashboard

### Request Parameters
1. Required:
- `prompt`: Text description of desired image edit
- `input_image`: Base64 encoded reference image (up to 20MB/20 megapixels)

2. Optional Parameters:
- `aspect_ratio`: Image dimensions (default 1:1, supports 3:7 to 7:3 ratios)
- `seed`: Integer for reproducible results
- `safety_tolerance`: Moderation level (0-2)
- `output_format`: "jpeg" or "png"
- `webhook_url`: Async completion notification
- `webhook_secret`: Webhook signature verification

### Request Flow
1. Submit edit request to endpoint
2. Receive `request_id` and `polling_url`
3. Poll `polling_url` until status is "Ready"
4. Retrieve result from `result['sample']` URL

### Key Capabilities
- Simple object modifications
- Character consistency across edits
- Text replacement within images
- Natural-looking contextual changes

### Constraints
- Signed result URLs valid for 10 minutes
- Images default to 1024x1024 pixels

## Files to Modify
1. `/openai_forward/flux/bfl_api.py` - Add FluxKontext class
2. `/openai_forward/base.py` - Update routing for image edits

## Current State Analysis
- `ImageEditPlatform.flux1_kontext` enum already exists
- Base routing detects `/images/edits` but currently only logs and forwards to OpenAI
- FLUX infrastructure (FluxBase, ContentModerationError) already implemented
- BFL_API_KEY environment variable support already exists