import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from fugashi import Tagger
from collections import defaultdict, Counter
import json
import re

class JapaneseTextAnalyzer:
    @staticmethod
    def create_kanji_grid(kanji_list, freq_dict):
        """Create a grid of kanji with their frequencies"""
        # Calculate grid dimensions
        n_kanji = len(kanji_list)
        n_cols = int(np.ceil(np.sqrt(n_kanji)))
        n_rows = int(np.ceil(n_kanji / n_cols))
        
        # Pad the list to fill the grid
        total_cells = n_rows * n_cols
        kanji_list = list(kanji_list) + [''] * (total_cells - len(kanji_list))
        
        # Create normalized frequencies for color mapping
        frequencies = np.array([freq_dict.get(k, 0) if k else 0 for k in kanji_list])
        if frequencies.max() > 0:
            normalized_freq = frequencies / frequencies.max()
        else:
            normalized_freq = frequencies
            
        return np.array(kanji_list).reshape(n_rows, n_cols), normalized_freq.reshape(n_rows, n_cols)
    
    def plot_kanji_maps(self, total_freq):
        """Create Kanji Maps divided by JLPT Level"""
        # Group kanji by JLPT level
        level_kanji = defaultdict(list)
        for kanji, freq in total_freq.items():
            level = self.kanji_dict.get(kanji, 'Unknown')
            level_kanji[level].append((kanji, freq))
        
        # Sort levels in order
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Create subplot grid
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        for idx, level in enumerate(level_order):
            kanji_freq = level_kanji[level]
            if not kanji_freq:
                continue
                
            # Sort by frequency
            kanji_freq.sort(key=lambda x: x[1], reverse=True)
            kanji_list = [k for k, _ in kanji_freq]
            freq_dict = dict(kanji_freq)
            
            # Create grid
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            kanji_grid, freq_grid = self.create_kanji_grid(kanji_list, freq_dict)
            
            # Create heatmap
            im = ax.imshow(freq_grid, cmap='Blues')
            
            # Add kanji text
            for i in range(len(kanji_grid)):
                for j in range(len(kanji_grid[i])):
                    if kanji_grid[i][j]:
                        color = 'white' if freq_grid[i][j] > 0.5 else 'black'
                        ax.text(j, i, kanji_grid[i][j], ha='center', va='center', color=color)
                        
            # Add frequency info in title
            n_kanji = len(kanji_list)
            ax.set_title(f'JLPT {level}\n({n_kanji} unique kanji)')
            ax.axis('off')
        
        plt.suptitle('Kanji Usage by JLPT Level', fontsize=16)
        plt.tight_layout()
        plt.show()
    def __init__(self):
        self.tagger = Tagger()
        self.kanji_dict = {}
        self.vocab_dict = {}
        self.text_data = None
        self.speaker_frequency = None
        self.kanji_pattern = re.compile(r'[一-龯]')  # Pattern for matching kanji
        
    def load_data(self, text_file, kanji_file, vocab_file):
        """Load all necessary data files"""
        # Load main text data
        self.text_data = pd.read_csv(text_file)
        
        # Load kanji JLPT data
        kanji_df = pd.read_csv(kanji_file)
        self.kanji_dict = dict(zip(kanji_df['kanji'], kanji_df['jlpt_level']))
        
        # Load vocabulary JLPT data
        vocab_df = pd.read_csv(vocab_file)
        self.vocab_dict = dict(zip(vocab_df['kanji'], vocab_df['jlpt_level']))
        
    def extract_kanji(self, text):
        """Extract individual kanji from text"""
        return self.kanji_pattern.findall(text)
    
    def get_kanji_frequency(self, text_series, speaker_series=None):
        """
        Calculate kanji frequency, optionally grouped by speaker
        Returns both overall frequencies and speaker-specific frequencies
        """
        total_freq = Counter()
        speaker_freq = defaultdict(Counter)
        
        for text, speaker in zip(text_series, speaker_series if speaker_series is not None else [''] * len(text_series)):
            kanji_list = self.extract_kanji(text)
            total_freq.update(kanji_list)
            if speaker:
                speaker_freq[speaker].update(kanji_list)
                
        return total_freq, speaker_freq
    
    def get_jlpt_distribution(self, frequency_counter):
        """Calculate JLPT level distribution for a frequency counter"""
        jlpt_dist = defaultdict(int)
        for kanji, freq in frequency_counter.items():
            jlpt_level = self.kanji_dict.get(kanji, 'Unknown')
            jlpt_dist[jlpt_level] += freq
        return dict(jlpt_dist)
    
    def analyze_text(self):
        """Perform complete analysis on loaded text data"""
        if self.text_data is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        # Get frequencies
        total_freq, speaker_freq = self.get_kanji_frequency(
            self.text_data['joined_japanese_text'],
            self.text_data['speaker']
        )
        
        # Store speaker_frequency for later use
        self.speaker_frequency = speaker_freq
        
        # Get JLPT distributions
        total_jlpt_dist = self.get_jlpt_distribution(total_freq)
        speaker_jlpt_dist = {
            speaker: self.get_jlpt_distribution(freq)
            for speaker, freq in speaker_freq.items()
        }
        
        return {
            'total_frequency': total_freq,
            'speaker_frequency': speaker_freq,
            'total_jlpt_distribution': total_jlpt_dist,
            'speaker_jlpt_distribution': speaker_jlpt_dist
        }
    
    def plot_jlpt_distribution(self, jlpt_dist, title="JLPT Level Distribution"):
        """Plot JLPT level distribution with ordered levels"""
        # Define the order of JLPT levels
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Create ordered data
        ordered_data = {level: jlpt_dist.get(level, 0) for level in level_order}
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'Level': list(ordered_data.keys()),
            'Count': list(ordered_data.values())
        })
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Level', y='Count', hue='Level', 
                   palette='Blues_r', legend=False)
        plt.title(title)
        plt.xlabel("JLPT Level")
        plt.ylabel("Frequency")
        plt.xticks(rotation=0)
        
        # Add value labels on top of bars
        for i, v in enumerate(ordered_data.values()):
            plt.text(i, v, str(v), ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()
    
    def plot_speaker_jlpt_distribution(self, speaker_jlpt_dist):
        """Plot JLPT distribution by speaker as a stacked marimekko chart"""
        # Convert to DataFrame
        data = []
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Calculate unique kanji count per speaker
        unique_counts = {}
        for speaker, dist in speaker_jlpt_dist.items():
            total_unique = sum(1 for freq in self.speaker_frequency[speaker].values() if freq > 0)
            unique_counts[speaker] = total_unique
            
        # Sort speakers by unique kanji count
        sorted_speakers = sorted(unique_counts.keys(), key=lambda x: unique_counts[x], reverse=True)
        
        # Prepare data
        for speaker in sorted_speakers:
            dist = speaker_jlpt_dist[speaker]
            for level in level_order:
                data.append({
                    'Speaker': speaker,
                    'JLPT Level': level,
                    'Count': dist.get(level, 0),
                    'Unique Kanji': unique_counts[speaker]
                })
        
        df = pd.DataFrame(data)
        
        # Create stacked bar chart
        plt.figure(figsize=(15, 8))
        
        # Create stacked bars
        bottom_vals = np.zeros(len(sorted_speakers))
        colors = sns.color_palette('Blues_r', n_colors=len(level_order))
        
        for i, level in enumerate(level_order):
            mask = df['JLPT Level'] == level
            values = df[mask]['Count'].values
            plt.bar(df[mask]['Speaker'], values, bottom=bottom_vals, 
                   label=level, color=colors[i])
            bottom_vals += values
        
        # Add total unique kanji count on top
        for i, speaker in enumerate(sorted_speakers):
            plt.text(i, bottom_vals[i], f'Unique: {unique_counts[speaker]}', 
                    ha='center', va='bottom')
        
        plt.title("JLPT Level Distribution by Speaker")
        plt.xlabel("Speaker")
        plt.ylabel("Kanji Count")
        plt.legend(title="JLPT Level", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


# Initialize analyzer
analyzer = JapaneseTextAnalyzer()

# Load data
analyzer.load_data(
    r"C:\Users\andre\Desktop\projects\shin_chan_coal_town\jpn_text_utils\shin_chan_coal_town_jp_text.csv",
    r"C:\Users\andre\Desktop\projects\shin_chan_coal_town\jpn_text_utils\kanji_jltp_tagged.txt",
    r"C:\Users\andre\Desktop\projects\shin_chan_coal_town\jpn_text_utils\vocab_jltp_tagged.txt"
)


# Perform analysis
results = analyzer.analyze_text()

# Plot Kanji Maps
analyzer.plot_kanji_maps(results['total_frequency'])

# Plot overall JLPT distribution
analyzer.plot_jlpt_distribution(results['total_jlpt_distribution'])

# Plot speaker-specific JLPT distribution
analyzer.plot_speaker_jlpt_distribution(results['speaker_jlpt_distribution'])

# Access raw frequency data
print("\nMost common kanji overall:")
for kanji, freq in results['total_frequency'].most_common(10):
    print(f"{kanji}: {freq}")

print("\nMost common kanji by speaker:")
for speaker, freq in results['speaker_frequency'].items():
    print(f"\n{speaker}:")
    for kanji, count in freq.most_common(5):
        print(f"{kanji}: {count}")