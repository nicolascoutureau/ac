# Vertical Stacking Feature for Multiple Speakers

## Overview

When multiple speakers (2-4 people) are detected in a scene and they're too far apart to fit in a vertical crop, the system automatically **stacks them vertically** instead of using letterboxing. This is perfect for:

- 📹 **Interviews** (interviewer + interviewee)
- 💬 **Conversations** (2-4 people talking)
- 🎙️ **Panel discussions** (2-4 panelists)
- 🎬 **Dialogue scenes** (multiple speakers)

## How It Works

### Strategy Selection Logic

```
Scene Analysis:
├─ 0 people detected → LETTERBOX (show full scene)
├─ 1 person detected → TRACK (follow person)
├─ Multiple people:
   ├─ People fit horizontally → TRACK (follow group)
   ├─ 2-4 people, too far apart → STACK (stack vertically) ⭐
   └─ 5+ people or complex → LETTERBOX (show full scene)
```

### Visual Examples by Aspect Ratio

**Original Frame (1920x1080):**

```
┌─────────────────────────────────────┐
│                                     │
│   👤 Person 1      👤 Person 2      │  ← Too far apart for single crop
│                                     │
└─────────────────────────────────────┘
```

#### 9:16 Vertical (Portrait)

**STACK Strategy: Vertical stacking**

```
┌───────────┐
│           │
│  👤 P1    │  ← Top half: Person 1
│           │
├───────────┤
│           │
│  👤 P2    │  ← Bottom half: Person 2
│           │
└───────────┘
```

#### 16:9 Horizontal (Landscape)

**STACK Strategy: Horizontal stacking**

```
┌──────────────────────┐
│          │           │
│  👤 P1   │   👤 P2   │  ← Side by side
│          │           │
└──────────────────────┘
```

#### 1:1 Square

**STACK Strategy: Grid layout**

2 People: Side by side

```
┌─────────────┐
│      │      │
│  👤  │  👤  │
│  P1  │  P2  │
└─────────────┘
```

4 People: 2x2 grid

```
┌─────────────┐
│  👤  │  👤  │ ← Top row
├──────┼──────┤
│  👤  │  👤  │ ← Bottom row
└─────────────┘
```

## Examples by Aspect Ratio

### 9:16 Vertical (TikTok, Reels, Shorts)

**2 People (Interview Style):**

- **Top 50%**: Person 1
- **Bottom 50%**: Person 2
- Each person gets full width, half height

**3 People (Panel Discussion):**

- **Top 33%**: Person 1
- **Middle 33%**: Person 2
- **Bottom 33%**: Person 3

**4 People (Group Conversation):**

- **Each gets 25%** of vertical space
- Sorted left-to-right → displayed top-to-bottom

### 16:9 Horizontal (YouTube, TV)

**2 People (News Split Screen):**

- **Left 50%**: Person 1
- **Right 50%**: Person 2
- Each person gets full height, half width

**3-4 People (Panel Discussion):**

- All displayed side-by-side
- Equal width for each person

### 1:1 Square (Instagram Posts)

**2 People:**

- Side by side (50% width each)
- Full height for both

**3 People:**

- Top row: 2 people (50% width, 50% height each)
- Bottom row: 1 person (full width, 50% height)

**4 People:**

- 2x2 grid
- Each person: 50% width, 50% height

## Implementation Details

### Function: `create_stacked_frame()`

```python
def create_stacked_frame(frame, people_data, output_width, output_height, aspect_ratio):
    """
    Creates an aspect-ratio-aware stacked frame with multiple people.

    Process:
    1. Detect aspect ratio:
       - < 0.8 (portrait): Stack vertically
       - > 1.2 (landscape): Stack horizontally
       - ~1.0 (square): Grid layout

    2. For each person:
       - Crop centered on their face/body
       - Resize to fit section
       - Place in appropriate position
    """
```

### Key Features

1. **Aspect Ratio Aware**: Automatically adapts layout based on output format

   - **Portrait (9:16)**: Vertical stacking (top to bottom)
   - **Landscape (16:9)**: Horizontal stacking (side by side)
   - **Square (1:1)**: Grid layout (2x2 for 4 people)

2. **Proper Aspect Ratio Maintained**: Each section maintains correct proportions

   - No stretching or distortion of people
   - Each section's crop calculated with proper aspect ratio
   - Resize is 1:1 (no stretching) after correct crop

3. **Dynamic Smart Zoom**: Each person gets optimally zoomed crop (no stretching)

   - Zoom calculated based on person size and section dimensions
   - Target: Person fills ~70% of their section
   - Adapts per person (1.3x to 2.5x range)
   - Avoids showing repetitive/overlapping areas
   - No stretching artifacts - crop matches section aspect ratio
   - Tighter framing on each individual person
   - Creates more distinct sections
   - Better engagement and clarity

4. **Smart Cropping**: Each person gets a centered crop based on their detected position

5. **Face Priority**: Uses face detection when available, falls back to person bounding box

6. **Equal Spacing**: Each person gets equal space in their dimension

7. **Sorted Display**: People displayed based on their original position (left-to-right)

## When It's Used

### ✅ STACK Strategy Applied When:

- 2-4 people detected in scene
- People are too far apart horizontally (don't fit in vertical aspect ratio)
- Better than letterboxing for conversational content

### ❌ STACK Not Used When:

- Only 1 person (uses TRACK)
- People close together horizontally (uses TRACK on group)
- 5+ people detected (uses LETTERBOX - too many to stack nicely)
- No people detected (uses LETTERBOX)

## Advantages Over Letterboxing

### Letterboxing (Old Behavior)

```
┌───────────┐
│▓▓▓▓▓▓▓▓▓▓▓│  ← Black bar
├───────────┤
│👤      👤 │  ← Full scene (people small)
├───────────┤
│▓▓▓▓▓▓▓▓▓▓▓│  ← Black bar
└───────────┘
```

❌ People appear small
❌ Wasted screen space
❌ Poor engagement on mobile

### Stacking (New Behavior)

```
┌───────────┐
│           │
│    👤     │  ← Person 1 (large, clear)
│           │
├───────────┤
│           │
│    👤     │  ← Person 2 (large, clear)
│           │
└───────────┘
```

✅ Each person clearly visible
✅ Uses full screen space
✅ Better for mobile/social media
✅ More engaging presentation

## Performance

### Processing Speed

- **Same as TRACK strategy** - no performance penalty
- Each section processed independently
- Efficient crop and resize operations

### Quality

- **Full resolution crops** for each person
- No quality loss from stacking
- Each person gets optimal framing

## Use Cases

### Perfect For:

- 🎥 **Podcast interviews** (host + guest)
- 📺 **News interviews** (anchor + guest)
- 💼 **Business presentations** (2-3 speakers)
- 🎓 **Educational videos** (teacher + student)
- 🎬 **Dialogue scenes** (2-4 characters)

### Not Ideal For:

- Large group shots (5+ people) → uses LETTERBOX
- Single person videos → uses TRACK
- People already close together → uses TRACK

## Configuration

### Automatic Configuration

No configuration needed! The system automatically:

1. Detects number of people
2. Calculates if they fit horizontally
3. Chooses STACK if 2-4 people are too far apart

### Manual Override

Currently not available, but could be added as a parameter:

```python
force_strategy="stack"  # Future feature
```

## Example Output Log

```
📋 Step 3: Generated Processing Plan
  - Scene 1 (00:00:00.000 -> 00:00:05.000): Found 2 person(s). Strategy: STACK
  - Scene 2 (00:00:05.000 -> 00:00:15.000): Found 1 person(s). Strategy: TRACK
  - Scene 3 (00:00:15.000 -> 00:00:25.000): Found 2 person(s). Strategy: TRACK
  - Scene 4 (00:00:25.000 -> 00:00:35.000): Found 6 person(s). Strategy: LETTERBOX
```

**Scene 1**: Two people far apart → **STACK** (split screen vertically)
**Scene 2**: One person → **TRACK** (follow person)
**Scene 3**: Two people close → **TRACK** (follow group)
**Scene 4**: Six people → **LETTERBOX** (show full scene)

## Limitations

1. **Maximum 4 People**: With 5+ people, stacking becomes too cramped
2. **Equal Space**: Each person gets equal vertical space (no custom ratios yet)
3. **No Overlays**: People are stacked, not overlaid (no picture-in-picture)
4. **Static Order**: Order based on horizontal position (can't customize)

## Future Enhancements

Possible improvements:

- [ ] Custom height ratios (e.g., 70% interviewer, 30% guest)
- [ ] Side-by-side option for 16:9 output
- [ ] Animated transitions between people
- [ ] Picture-in-picture mode
- [ ] Smart reordering based on who's speaking
- [ ] Optional borders between stacked sections

## Summary

The **STACK** strategy provides an intelligent way to handle multiple speakers in vertical video format, ensuring each person is clearly visible without wasting screen space. It's automatically applied when appropriate, making conversation videos much more engaging on mobile devices and social media platforms.

🎯 **Result**: Better viewer engagement and clearer communication in multi-person videos!
