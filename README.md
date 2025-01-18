# Shin Chan: Shiro and the Coal Town - Japanese Text Dump

## Overview
This project is a set of utils to **extract**, **process** and **analyze** the Japanese text from the game "**Shin Chan: Shiro and the Coal Town**". 
<br /><br />The output is provided as a CSV file containing game dialogues with speaker identification, translations, and supplementary information for Japanese language learners. See the **release section of the right**.
<br /><br />Note that **this may not be a complete text dump of the game**. The extraction process relies on capturing text while playing the game, which means coverage is not systematic and the total percentage of extracted text cannot be determined. While most steps (screenshot capture, deduplication, and organization) have been automated, some data may have been lost during processing.
<br /><br />Despite these limitations, the dataset should serve as a valuable reference for learners, containing over 2,000 text extractions and approximately 4,500 speaker-tagged sentences.

All rights to the [developers](https://game.neoscorp.jp/shinchan_coaltown/index_en.html).

## Project Structure
```
shin_chan_coal_town/
├── image_to_text_utils/ 
│   ├── screenshots_to_text/            # Screenshots to be sent to Claude
│   ├── image_to_text.py                # Code to perform API calls to Claude for OCR and translations from images
│   └── text_utils.py                   # Utils to merge Json files and deduplicate text entries

├── jpn_text_utils/
│   ├── ipynb_checkpoints/
│   ├── __pycache__/
│   ├── japanese_text_analysis.py       # Functions to perform basic analytics on Japanese text
│   ├── kanji_jltp_tagged               # Kanji tagged to JLPT level
│   ├── README_JLPT.INFO                # Source of JLPT info used in the project
│   ├── shin_chan_coal_town_jp.text     # Japanese text dump used in the scripts
│   ├── text_analysis.ipynb             # A display of the charts created through <japanese_text_analysis.py>
│   └── vocab_jltp_tagged               # Vocabulary tagged to JLPT level
└── screen_capture_utils/
    ├── characters_catalogue/           # Visual archive of the characters in the game
    ├── screenshots/                    # Location where screenshots are saved while playing the game
    ├── templates/                      # Used for detection of dialogue boxes
    ├── screen_capture_debugger.py      # App + debugger used to set parameters
    ├── screen_capture_GPU_enabled.py   # App to take screenshots of the dialogues with GPU support
    └── screenshots_organizer.py        # App to dedupe screenshots and organize them in folder for easier processing
```

## Components

### 1. Screen Capture Utils
- Located in `screen_capture_utils/`
- **Purpose**: Captures dialogue boxes during gameplay
- **Key Features**:
  - Real-time dialogue box detection
  - GPU-enabled capture capabilities
  - Screenshot organization and management
  - Character-specific template matching
  - Debug mode to fine tune screencapture (params setting)

### 2. Image to Text Utils
- Located in `image_to_text_utils/`
- **Purpose**: Converts captured screenshots to text
- **Key Features**:
  - Image preprocessing for OCR
  - Batch processing, text extraction via Claude API
  - Screenshot deduplication

### 3. Japanese Text Utils
- Located in `jpn_text_utils/`
- **Purpose**: Analyzes extracted Japanese text
- **Key Features**:
  - JLPT level classification
  - Vocabulary and kanji tagging
  - Language complexity analysis
  - Basic charting and insights in this [Jupyter Notebook](https://github.com/andrebvq/shin_chan_coal_town/blob/main/jpn_text_utils/text_analysis.ipynb)

## Workflow
1. **Capture Phase**
   - Game runs with screen capture utils active
   - Dialogue boxes are detected and captured
   - Screenshots are organized and deduplicated

2. **Text Extraction Phase**
   - Screenshots are processed in batches
   - Claude API performs OCR and translation
   - Results are collected and structured

3. **Analysis Phase**
   - Text is tagged with JLPT levels
   - Language complexity metrics are calculated
   - Results are compiled into final CSV format

## Output
The main output is a CSV file containing:
- Original Japanese text
- Speaker identification
- English translations
- JLPT level classifications
- Additional translation notes
