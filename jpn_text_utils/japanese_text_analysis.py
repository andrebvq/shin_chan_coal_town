import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import numpy as np
from fugashi import Tagger
from collections import defaultdict, Counter
import re

class JapaneseTextAnalyzer:
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
        """Calculate kanji frequency, optionally grouped by speaker"""
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
    
    def plot_kanji_maps(self, total_freq):
        """Create Kanji Maps divided by JLPT Level with frequency legend"""
        try:
            fp = FontProperties(family='MS Gothic')
        except:
            fp = FontProperties(family='DFKai-SB')  # Fallback font
            
        # Group kanji by JLPT level
        level_kanji = defaultdict(list)
        for kanji, freq in total_freq.items():
            level = self.kanji_dict.get(kanji, 'Unknown')
            level_kanji[level].append((kanji, freq))
        
        # Sort levels in order
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Create subplot grid with space for legend
        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 0.2])
        
        # Define frequency bins for colors
        freq_bins = [1, 5, 10, 25, 50, 100]
        colors = plt.cm.Blues(np.linspace(0.2, 1, len(freq_bins)))
        
        for idx, level in enumerate(level_order):
            if idx >= 6:  # Skip if more than 6 levels (2x3 grid)
                continue
                
            kanji_freq = level_kanji[level]
            if not kanji_freq:
                continue
            
            # Sort by frequency
            kanji_freq.sort(key=lambda x: x[1], reverse=True)
            kanji_list = [k for k, _ in kanji_freq]
            freq_dict = dict(kanji_freq)
            
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            
            # Calculate grid dimensions
            n_kanji = len(kanji_list)
            n_cols = int(np.ceil(np.sqrt(n_kanji)))
            n_rows = int(np.ceil(n_kanji / n_cols))
            
            # Create color grid
            color_grid = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                for j in range(n_cols):
                    idx_linear = i * n_cols + j
                    if idx_linear < len(kanji_list):
                        freq = freq_dict[kanji_list[idx_linear]]
                        color_grid[i, j] = freq
            
            # Normalize color grid
            if color_grid.max() > 0:
                color_grid = color_grid / color_grid.max()
            
            # Plot heatmap
            im = ax.imshow(color_grid, cmap='Blues')
            
            # Add kanji text
            for i in range(n_rows):
                for j in range(n_cols):
                    idx_linear = i * n_cols + j
                    if idx_linear < len(kanji_list):
                        kanji = kanji_list[idx_linear]
                        freq = freq_dict[kanji]
                        color = 'white' if color_grid[i, j] > 0.5 else 'black'
                        ax.text(j, i, kanji, ha='center', va='center', 
                               color=color, fontproperties=fp, fontsize=10)
            
            ax.set_title(f'JLPT {level}\n({len(kanji_list)} unique kanji)')
            ax.axis('off')
        
        # Add color legend
        ax_legend = fig.add_subplot(gs[-1, :])
        ax_legend.axis('off')
        
        # Create legend
        legend_elements = []
        for i in range(len(freq_bins)-1):
            label = f'{freq_bins[i]}+ occurrences'
            if i == len(freq_bins)-2:
                label = f'{freq_bins[i]}+ occurrences'
            patch = plt.Rectangle((0, 0), 1, 1, facecolor=colors[i])
            legend_elements.append((patch, label))
        
        patches, labels = zip(*legend_elements)
        ax_legend.legend(patches, labels, loc='center', ncol=len(freq_bins)-1,
                        bbox_to_anchor=(0.5, 0.5))
        
        plt.suptitle('Kanji Usage by JLPT Level', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_sentence_statistics(self):
        """Create dot plots for sentence statistics by speaker"""
        if self.text_data is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        # Calculate the metrics
        stats_df = self.text_data.groupby('speaker').agg({
            'joined_japanese_text': 'count',  # Total number of sentences
            'text_length': 'mean'  # Average length already in characters
        }).reset_index()
        
        # Rename columns for clarity
        stats_df.columns = ['speaker', 'total_sentences', 'avg_length']
        
        # Sort by total sentences descending
        stats_df = stats_df.sort_values('total_sentences', ascending=False)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        plt.subplots_adjust(wspace=0.1)
        
        # Define the metrics to plot
        metrics = ['total_sentences', 'avg_length']
        titles = ['Tot # Sentences by Speaker', 'Avg Sentence Length by # Chars']
        
        # Plot each metric
        for ax, metric, title in zip(axes, metrics, titles):
            sns.scatterplot(data=stats_df, 
                          y='speaker',
                          x=metric,
                          ax=ax,
                          alpha=0.8,
                          color='#1f77b4',  # Darker blue
                          s=75)  # Reduced dot size
            
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlabel('Count')
            
            # Only show y-axis labels on the first plot
            if ax != axes[0]:
                ax.set_ylabel('')
            
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nSpeaker Statistics:")
        print(stats_df.round(2).to_string(index=False))
        
        
    def _calc_jlpt_weights(self, text):
        """Calculate JLPT-weighted score for a text"""
        kanji_list = self.extract_kanji(text)
        weights = {'N1': 5, 'N2': 4, 'N3': 3, 'N4': 2, 'N5': 1, 'Unknown': 3}
        score = 0
        total_kanji = len(kanji_list)
        
        if total_kanji == 0:
            return 0
            
        for kanji in kanji_list:
            level = self.kanji_dict.get(kanji, 'Unknown')
            score += weights[level]
            
        return (score / (total_kanji * 5)) * 30  # JLPT component max 30 points

    def _calc_length_factor(self, text):
        """Calculate length-based complexity score"""
        # Exponentially increasing score up to a cap
        base_length = 20  # Standard sentence length
        max_score = 20    # Maximum points for length
        
        length = len(text)
        score = min((length / base_length) ** 1.5, 2) * (max_score / 2)
        return min(score, max_score)

    def _calc_grammar_complexity(self, text):
        """Calculate grammatical complexity score"""
        max_score = 25  # Maximum points for grammar
        score = 0
        
        # Complex grammar patterns
        patterns = {
            'causative': ['させ', 'せる'],
            'passive': ['れる', 'られる'],
            'conditional': ['たら', 'ば', 'なら', 'と'],
            'honorific': ['お', 'ご', 'です', 'ます'],
            'conjunctions': ['ので', 'のに', 'けど', 'が'],
            'complex_forms': ['っぽい', 'そう', 'よう', 'みたい'],
            'formal': ['である', 'にて', 'により'],
            'emphatic': ['こそ', 'さえ', 'すら', 'だけ'],
            'compound': ['ながら', 'つつ', 'にも関わらず']
        }
        
        # Weights for different pattern types
        weights = {
            'causative': 3,
            'passive': 2.5,
            'conditional': 2,
            'honorific': 1.5,
            'conjunctions': 1,
            'complex_forms': 2,
            'formal': 2.5,
            'emphatic': 2,
            'compound': 3
        }
        
        for pattern_type, patterns_list in patterns.items():
            for pattern in patterns_list:
                if pattern in text:
                    score += weights[pattern_type]
        
        return min(score, max_score)

    def _calc_vocabulary_diversity(self, text):
        """Calculate vocabulary diversity score"""
        max_score = 15  # Maximum points for vocabulary diversity
        
        # Tokenize text
        words = [token.surface for token in self.tagger(text)]
        unique_words = set(words)
        
        # Calculate diversity ratio with diminishing returns
        diversity_ratio = len(unique_words) / max(len(words), 1)
        score = (diversity_ratio ** 0.5) * max_score
        
        return min(score, max_score)

    def _calc_specialized_vocab(self, text):
        """Calculate specialized vocabulary score"""
        max_score = 10  # Maximum points for specialized vocabulary
        score = 0
        
        # Define specialized vocabulary patterns
        specialized = {
            'dialect': ['やん', 'せん', 'へん', 'わ', 'だす', 'ばい'],
            'slang': ['めっちゃ', 'すげー', 'やべー', 'まじ', 'わず'],
            'formal_business': ['致します', '申し上げる', '参る', '存じ'],
            'technical': ['における', 'に関する', 'に基づく', '前述の'],
            'literary': ['かくて', 'されど', 'ごとき', 'いかん']
        }
        
        weights = {
            'dialect': 1.5,
            'slang': 1,
            'formal_business': 2,
            'technical': 2.5,
            'literary': 2
        }
        
        for vocab_type, patterns in specialized.items():
            for pattern in patterns:
                if pattern in text:
                    score += weights[vocab_type]
        
        return min(score, max_score)

    def compute_complexity(self):
        """Compute overall sentence complexity scores"""
        if self.text_data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Calculate complexity scores for each sentence
        complexities = []
        
        for text in self.text_data['joined_japanese_text']:
            # Calculate individual components
            jlpt_score = self._calc_jlpt_weights(text)
            length_score = self._calc_length_factor(text)
            grammar_score = self._calc_grammar_complexity(text)
            vocab_div_score = self._calc_vocabulary_diversity(text)
            spec_vocab_score = self._calc_specialized_vocab(text)
            
            # Calculate total score (max 100)
            total_score = jlpt_score + length_score + grammar_score + vocab_div_score + spec_vocab_score
            total_score = min(total_score, 100)
            
            complexities.append({
                'total_score': total_score,
                'jlpt_score': jlpt_score,
                'length_score': length_score,
                'grammar_score': grammar_score,
                'vocab_diversity_score': vocab_div_score,
                'specialized_vocab_score': spec_vocab_score
            })
        
        # Add scores to dataframe
        complexity_df = pd.DataFrame(complexities)
        self.text_data = pd.concat([self.text_data, complexity_df], axis=1)
        
        # Calculate average scores by speaker
        self.speaker_scores = self.text_data.groupby('speaker').agg({
            'total_score': 'mean',
            'jlpt_score': 'mean',
            'length_score': 'mean',
            'grammar_score': 'mean',
            'vocab_diversity_score': 'mean',
            'specialized_vocab_score': 'mean'
        }).round(2)
        
    def plot_complexity_analysis(self):
        """Visualize complexity components by speaker"""
        if 'total_score' not in self.text_data.columns:
            self.compute_complexity()
            
        if not hasattr(self, 'speaker_scores'):
            raise ValueError("No complexity scores found. Run compute_complexity first.")
            
        # Sort speakers by total score and get component data
        speaker_scores = self.speaker_scores.sort_values('total_score', ascending=True)
        component_data = speaker_scores.drop('total_score', axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot stacked bars
        component_data.plot(kind='barh', 
                          stacked=True,
                          ax=ax,
                          width=0.8,
                          cmap='Blues')
        
        # Customize plot
        ax.set_title('Sentence Complexity Components by Speaker')
        ax.set_xlabel('Score Contribution')
        ax.set_ylabel('Speaker')
        ax.legend(bbox_to_anchor=(1.05, 1), 
                 loc='upper left',
                 title='Components')
        
        plt.tight_layout()
        plt.show()
        
        # Print scores
        print("\nComplexity Scores by Component:")
        print(component_data.round(2).to_string())        
        
    def plot_jlpt_distribution(self, jlpt_dist, title="JLPT Level Distribution"):
        """Plot JLPT level distribution showing both total and unique counts"""
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Calculate total and unique counts
        total_counts = []
        unique_counts = []
        for level in level_order:
            # Total occurrences
            total_counts.append(jlpt_dist.get(level, 0))
            # Unique kanji
            unique_kanji = set(k for k, l in self.kanji_dict.items() 
                             if l == level and k in self.speaker_frequency)
            unique_counts.append(len(unique_kanji))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot bars
        x = np.arange(len(level_order))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, total_counts, width, label='Total Occurrences',
                       color='steelblue')
        rects2 = ax.bar(x + width/2, unique_counts, width, label='Unique Kanji',
                       color='lightsteelblue')
        
        # Add labels and titles
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(level_order)
        ax.legend()
        
        # Add value labels on bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                          xy=(rect.get_x() + rect.get_width()/2, height),
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        plt.show()


    def plot_speaker_jlpt_distribution(self, speaker_jlpt_dist):
        """Plot JLPT distribution by speaker as percentages with unique count line"""
        level_order = ['N5', 'N4', 'N3', 'N2', 'N1', 'Unknown']
        
        # Calculate percentages and unique counts
        data = []
        unique_counts = {}
        
        for speaker in speaker_jlpt_dist.keys():
            dist = speaker_jlpt_dist[speaker]
            total = sum(dist.values())
            unique_count = len(set(kanji for kanji in self.speaker_frequency[speaker].keys()))
            unique_counts[speaker] = unique_count
            
            for level in level_order:
                data.append({
                    'Speaker': speaker,
                    'JLPT Level': level,
                    'Percentage': (dist.get(level, 0) / total * 100) if total > 0 else 0
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Sort speakers by unique kanji count
        sorted_speakers = sorted(unique_counts.keys(), 
                               key=lambda x: unique_counts[x],
                               reverse=True)
        
        # Create plot with two y-axes
        fig, ax1 = plt.subplots(figsize=(15, 8))
        
        # Plot stacked percentage bars
        bottom_vals = np.zeros(len(sorted_speakers))
        colors = plt.cm.Blues(np.linspace(0.2, 0.8, len(level_order)))
        
        for i, level in enumerate(level_order):
            mask = df['JLPT Level'] == level
            values = []
            for speaker in sorted_speakers:
                speaker_data = df[(df['Speaker'] == speaker) & (df['JLPT Level'] == level)]
                values.append(speaker_data['Percentage'].iloc[0] if not speaker_data.empty else 0)
            
            ax1.bar(sorted_speakers, values, bottom=bottom_vals, 
                   label=level, color=colors[i])
            bottom_vals += values
        
        # Add unique count line on secondary y-axis
        ax2 = ax1.twinx()
        unique_values = [unique_counts[speaker] for speaker in sorted_speakers]
        ax2.plot(range(len(sorted_speakers)), unique_values, 'r-', linewidth=2, 
                 marker='o', label='Unique Kanji')
        
        # Customize axes
        ax1.set_title("JLPT Level Distribution by Speaker")
        ax1.set_xlabel("Speaker")
        ax1.set_ylabel("Percentage")
        ax2.set_ylabel("Unique Kanji Count", color='r')
        
        # Set tick labels
        plt.xticks(range(len(sorted_speakers)), sorted_speakers, rotation=45, ha='right')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  bbox_to_anchor=(1.15, 1), loc='upper left')
        
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

analyzer.plot_sentence_statistics()


analyzer.compute_complexity()
analyzer.plot_complexity_analysis()

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