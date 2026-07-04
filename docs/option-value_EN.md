# Option Guide
This guide explains recommended values for blurring Person and Face objects during outdoor streaming to protect the privacy of passersby.

> [!IMPORTANT]
> This filter adds a default delay of **0.5 seconds** for more stable tracking.

> [!TIP]
> Clicking `Defaults` at the bottom of the filter panel resets all settings to the recommended Person values.

### Basic Settings
- **Model**
  - Recommended: Person `Medium` or `Large (Accurate)`, Face `YOLOv8n-Face (Face Detection, Accurate)` or higher
- **Object Category**
  - Recommended: `Person`, `Face`
- **Confidence Threshold**
  - Only objects with a confidence score above this value will be detected.
  - Recommended: Person `0.45`, Face `0.35`
- **Minimum Area Threshold**
  - Objects smaller than this area will be ignored.
  - Recommended: Person `3000`, Face `50`

### Masking
- **Masking Type**
  - Recommended: `Blur` or `Pixelate`
- **Intensity**
  - Recommended: `10`
- **Edge Feather**
  - Recommended: `30`
- **Mask Dynamic Expansion**
  - Expands the masking area proportionally to the detected object size.
  - Recommended: `Enabled`
    - **Dynamic Expansion Base Size (Pixels)**
      - The base expansion size for the object.
      - Recommended: Person `1.00`, Face `2.00`
    - **Dynamic Expansion Ratio**
      - The larger the object, the more the mask expands proportionally.
        - Example: When a face is far away it is small and fully covered, but up close the hair and ears may be exposed — so the mask scales up accordingly.
      - Recommended: Person `0.05`, Face `0.25`

### Continuous Tracking
- **Minimum Tracking Frames**
  - Sets how many consecutive frames an object must be detected before masking begins.
    - If an object is detected for fewer frames than this value, it will not be masked. (Brief misses are ignored.)
  - Recommended: `5`
- **Ignore Minimum Tracking Frames Size Threshold**
  - Objects larger than this value bypass the Minimum Tracking Frames setting and are masked immediately.
    - This ensures people who appear and disappear faster than the minimum frame count are still masked.
  - Recommended: Person `4.00`, Face `0.80`
- **Max Unseen Frames**
  - If tracking is lost, masking is maintained for this many frames. If the object is not found again within this limit, masking is removed.
  - Recommended: `15`
- **IoU Threshold (Overlap Tolerance)**
  - Represents the degree of overlap between objects detected in consecutive frames. Objects with an IoU below this value are treated as separate objects.
  - Recommended: Person `0.19`, Face `0.01`
- **Missed Object Recovery Distance Multiplier**
  - If tracking is completely lost due to no overlap, the search area is expanded proportionally to the object size to find the object again.
    - This helps re-identify the same person or face even if they move very quickly.
  - Recommended: Person `1.80`, Face `2.50`
- **Missed Object Recovery Size Ratio Limit**
  - Within the recovery search area, an object is only considered the same if its size differs from the lost object by less than this ratio.
  - Recommended: `1.70`

### Exclude Specific Targets from Masking (Face Recognition)
> For a detailed guide, see [here](FACE_GUIDE_EN.md).
- **Person Category**
  - If tracking multiple object types, specify which category represents a person.
  - Not required if your objects are already set to Person or Face.
- **Face Image Folder**
  - Select the folder containing reference face photos.
- **Face Match Similarity Threshold**
  - Faces with a similarity score above this value will be excluded from masking.
  - Recommended: `0.60`
- **Minimum Object Size for Recognition**
  - The minimum object size required to perform face recognition.
    - **This refers to the size of the object in the non-face object category!**
  - This option helps prevent false matches with distant passersby who are too far away.
  - Recommended: Person `19.00`, Face `0.60`
- **Face Recognition Check Interval (Frames)**
  - Face recognition is performed every N frames.
  - A shorter interval is fine if performance allows, but be aware that another person's similarity score could spike above the threshold momentarily.
  - Recommended: `5`-`15` (at 30fps)
- **Max Exempt Objects Limit**
  - Set the total number of people you want to exclude from masking.
  - `0` means unlimited.