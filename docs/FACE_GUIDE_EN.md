# Reference Photo Guide for Face Recognition

This guide helps you get the most out of the "Exclude Specific Targets from Masking (Face Recognition)" feature in the `detect` plugin.

When selecting a folder of your face photos, preparing multiple images using the tips below will significantly improve recognition accuracy.

## Before You Start
1. **Multiple photos and multiple people** are both fine.
2. The background of the photos does not matter.
3. What matters most is that your facial features are clearly visible — not your hairstyle.
4. **Face angle** has a big impact on recognition accuracy. Try to prepare at least 10 photos from a variety of angles.

## ✅ Recommended Photo Composition

For the best recognition rate, collect 10 or more photos into a single folder. (Selfies from your smartphone or captures from your stream work great.)

### 1. Varied Lighting Conditions
Reflect the lighting conditions you typically experience during outdoor streaming.
- A photo taken with **all room lights on (very bright)**
- A photo taken with all room lights off, lit only by outdoor light

### 2. Varied Face Angles and Expressions
Since you don't always face the camera directly, slightly turned photos are helpful.
- Straight-on (front-facing) photo
- Front-facing photo with mouth open, as if speaking naturally
- Side profile photos with head turned approximately 20 degrees left/right/up/down
- Photos at the angles your face typically appears during your stream

## ❌ Photos to Avoid (Causes of Lower Recognition Rate)

- **Photos where your face is too small**
  - If the face is too small (e.g., in a full-body shot) and the resolution breaks down, the AI may fail to locate facial landmarks (eyes, nose, mouth). Upper-body photos are recommended.
- **Photos with unclear facial features**
  - Avoid photos where thick-rimmed glasses obscure your eyes, long bangs completely cover your eyebrows and eyes, or you are wearing a face mask.
- **Heavy photo filters or retouching**
  - The AI analyzes the unique distance ratios between your actual facial features. If you use a filter that greatly enlarges your eyes or slims your jawline, the similarity score against your real face during streaming may be lower.
- **Photos with multiple people**
  - If two or more faces are detected in a single photo, there is a risk that features from the wrong person will be stored.

## 🛠️ Setup Instructions
1. Following the guide above, gather your photos into a single folder.
2. Open the `detect` filter settings panel in OBS.
3. Check the **"Exclude Specific Targets from Masking (Face Recognition)"** option.
4. Under **"Face Image Folder"**, select the folder you just created.
5. Test it yourself and fine-tune settings such as the similarity threshold.
    - Enable **`Show Face Similarity Stats`** and **`Show Face Similarity Score in OBS Log`**, then check whether your face consistently scores 0.6 or above.
      - If the score is too low, add more photos following the guide above, or try lowering the `Face Match Similarity Threshold`.
      - Once testing is complete, disable those options again.
6. Done!

> [!TIP]
> The default values are optimized for the **'Person'** object category.

## Option Descriptions
- **Face Match Similarity Threshold**
  - Faces with a similarity score above this value will be excluded from masking.
  - Typically, other people score 0.5 or below, while your own face scores 0.6 or above.
  - Make sure to enable **"Show Face Similarity Stats"** and **"Show Face Similarity Score in OBS Log"** in the Debug section and test with your own face!
- **Minimum Object Size for Recognition**
  - The minimum object size required to perform face recognition.
    - **This refers to the size of the object in the non-face object category!**
  - This option helps prevent false matches with distant passersby who are too far away.
- **Face Recognition Check Interval (Frames)**
  - Face recognition is performed every N frames.
  - A shorter interval is fine if performance allows, but be aware that another person's similarity score could spike above the threshold momentarily.
- **Max Exempt Objects Limit**
  - Set the total number of people you want to exclude from masking.
  - `0` means unlimited.

## Recommended Values
### Person Object
- **Face Match Similarity Threshold**: 0.60
- **Minimum Object Size for Recognition**: 15.00
- **Face Recognition Check Interval (Frames)**: 15
- **Max Exempt Objects Limit**: *(Set this to the total number of people you want to exclude. 0 means unlimited.)*
### Face Object
- **Face Match Similarity Threshold**: 0.60
- **Minimum Object Size for Recognition**: 0.60
- **Face Recognition Check Interval (Frames)**: 15
- **Max Exempt Objects Limit**: *(Set this to the total number of people you want to exclude. 0 means unlimited.)*
