#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import signal
import argparse
import os
import sys

class AudioAnalyzer:
    def __init__(self, file1_path, file2_path):
        """Initialize analyzer with two audio files"""
        self.file1_path = file1_path
        self.file2_path = file2_path
        
        # Load audio files
        self.sample_rate1, self.audio1 = self.load_audio(file1_path)
        self.sample_rate2, self.audio2 = self.load_audio(file2_path)
        
        # Verify sample rates match
        if self.sample_rate1 != self.sample_rate2:
            print(f"Warning: Sample rates don't match ({self.sample_rate1} vs {self.sample_rate2})")
        
        self.sample_rate = self.sample_rate1
        
        # Convert to mono for analysis (average channels)
        self.mono1 = self.to_mono(self.audio1)
        self.mono2 = self.to_mono(self.audio2)
        
        print(f"Loaded files:")
        print(f"  File 1: {os.path.basename(file1_path)} - {len(self.mono1)} samples, {self.sample_rate1} Hz")
        print(f"  File 2: {os.path.basename(file2_path)} - {len(self.mono2)} samples, {self.sample_rate2} Hz")

    def load_audio(self, file_path):
        """Load WAV file and return sample rate and audio data"""
        try:
            sample_rate, audio = wavfile.read(file_path)
            
            # Convert to float32 for processing
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype == np.uint8:
                audio = (audio.astype(np.float32) - 128.0) / 128.0
            
            return sample_rate, audio
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            sys.exit(1)

    def to_mono(self, audio):
        """Convert stereo to mono by averaging channels"""
        if len(audio.shape) == 2:
            return np.mean(audio, axis=1)
        return audio

    def calculate_spectrum(self, audio, nperseg=8192):
        """Calculate power spectrum using Welch's method"""
        frequencies, power = signal.welch(
            audio, 
            fs=self.sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg//2,
            scaling='density'
        )
        
        # Convert to dB
        power_db = 10 * np.log10(power + 1e-12)  # Add small value to avoid log(0)
        
        return frequencies, power_db

    def calculate_spectrogram(self, audio, nperseg=2048):
        """Calculate spectrogram for time-frequency analysis"""
        frequencies, times, spectrogram = signal.spectrogram(
            audio,
            fs=self.sample_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg//2,
            scaling='density'
        )
        
        # Convert to dB
        spectrogram_db = 10 * np.log10(spectrogram + 1e-12)
        
        return frequencies, times, spectrogram_db

    def create_comparison_plot(self, save_path=None):
        """Create comprehensive comparison plots"""
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Calculate spectra and spectrograms
        freq1, power1 = self.calculate_spectrum(self.mono1)
        freq2, power2 = self.calculate_spectrum(self.mono2)
        
        spec_freq1, spec_time1, spec1 = self.calculate_spectrogram(self.mono1)
        spec_freq2, spec_time2, spec2 = self.calculate_spectrogram(self.mono2)
        
        # 1. Average Frequency Response Comparison (Main Plot)
        ax1 = plt.subplot(3, 2, (1, 2))  # Top row, spans both columns
        
        # Plot frequency responses
        ax1.semilogx(freq1, power1, 'b-', linewidth=2, label=f'Original: {os.path.basename(self.file1_path)}')
        ax1.semilogx(freq2, power2, 'r-', linewidth=2, label=f'Processed: {os.path.basename(self.file2_path)}')
        
        # Calculate and plot difference
        # Interpolate to common frequency grid for difference calculation
        common_freq = freq1  # Use first file's frequency grid
        power2_interp = np.interp(common_freq, freq2, power2)
        difference = power2_interp - power1
        
        ax1_diff = ax1.twinx()
        ax1_diff.semilogx(common_freq, difference, 'g--', linewidth=1, alpha=0.7, label='Difference (Processed - Original)')
        ax1_diff.set_ylabel('Difference (dB)', color='g')
        ax1_diff.tick_params(axis='y', labelcolor='g')
        
        # Format main frequency plot
        ax1.set_xlim(20, 20000)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density (dB/Hz)')
        ax1.set_title('Frequency Response Comparison (20Hz - 20kHz)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        ax1_diff.legend(loc='upper right')
        
        # Add frequency markers
        freq_markers = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for fm in freq_markers:
            if 20 <= fm <= 20000:
                ax1.axvline(x=fm, color='gray', linestyle=':', alpha=0.5)
        
        # 2. Spectrogram - Original
        ax2 = plt.subplot(3, 2, 3)
        
        # Limit spectrogram to audible range
        freq_mask1 = spec_freq1 <= 20000
        im1 = ax2.pcolormesh(spec_time1, spec_freq1[freq_mask1], spec1[freq_mask1, :], 
                            shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
        ax2.set_yscale('log')
        ax2.set_ylim(20, 20000)
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title(f'Spectrogram: {os.path.basename(self.file1_path)}', fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax2)
        cbar1.set_label('Power (dB)')
        
        # 3. Spectrogram - Processed
        ax3 = plt.subplot(3, 2, 4)
        
        freq_mask2 = spec_freq2 <= 20000
        im2 = ax3.pcolormesh(spec_time2, spec_freq2[freq_mask2], spec2[freq_mask2, :], 
                            shading='gouraud', cmap='viridis', vmin=-80, vmax=-20)
        ax3.set_yscale('log')
        ax3.set_ylim(20, 20000)
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_xlabel('Time (s)')
        ax3.set_title(f'Spectrogram: {os.path.basename(self.file2_path)}', fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax3)
        cbar2.set_label('Power (dB)')
        
        # 4. Statistical Analysis
        ax4 = plt.subplot(3, 2, 5)
        
        # Calculate RMS levels in frequency bands
        bands = {
            'Sub Bass (20-60Hz)': (20, 60),
            'Bass (60-250Hz)': (60, 250),
            'Low Mid (250Hz-2kHz)': (250, 2000),
            'High Mid (2-8kHz)': (2000, 8000),
            'Treble (8-20kHz)': (8000, 20000)
        }
        
        band_names = list(bands.keys())
        levels1 = []
        levels2 = []
        
        for band_name, (low, high) in bands.items():
            # Find frequency indices for this band
            mask1 = (freq1 >= low) & (freq1 <= high)
            mask2 = (freq2 >= low) & (freq2 <= high)
            
            # Calculate average level in this band
            if np.any(mask1):
                level1 = np.mean(power1[mask1])
                levels1.append(level1)
            else:
                levels1.append(-80)
                
            if np.any(mask2):
                level2 = np.mean(power2[mask2])
                levels2.append(level2)
            else:
                levels2.append(-80)
        
        x_pos = np.arange(len(band_names))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, levels1, width, label='Original', alpha=0.8, color='blue')
        bars2 = ax4.bar(x_pos + width/2, levels2, width, label='Processed', alpha=0.8, color='red')
        
        ax4.set_xlabel('Frequency Bands')
        ax4.set_ylabel('Average Level (dB)')
        ax4.set_title('Frequency Band Analysis', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(band_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax4.text(bar1.get_x() + bar1.get_width()/2., height1 + 1,
                    f'{height1:.1f}', ha='center', va='bottom', fontsize=8)
            ax4.text(bar2.get_x() + bar2.get_width()/2., height2 + 1,
                    f'{height2:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Audio Statistics
        ax5 = plt.subplot(3, 2, 6)
        ax5.axis('off')
        
        # Calculate statistics
        rms1 = np.sqrt(np.mean(self.mono1**2))
        rms2 = np.sqrt(np.mean(self.mono2**2))
        peak1 = np.max(np.abs(self.mono1))
        peak2 = np.max(np.abs(self.mono2))
        
        # Convert to dB
        rms1_db = 20 * np.log10(rms1 + 1e-12)
        rms2_db = 20 * np.log10(rms2 + 1e-12)
        peak1_db = 20 * np.log10(peak1 + 1e-12)
        peak2_db = 20 * np.log10(peak2 + 1e-12)
        
        stats_text = f"""
Audio Statistics:

Original File:
• RMS Level: {rms1_db:.1f} dB
• Peak Level: {peak1_db:.1f} dB
• Duration: {len(self.mono1)/self.sample_rate:.1f} seconds
• Sample Rate: {self.sample_rate} Hz

Processed File:
• RMS Level: {rms2_db:.1f} dB
• Peak Level: {peak2_db:.1f} dB
• Duration: {len(self.mono2)/self.sample_rate:.1f} seconds
• Sample Rate: {self.sample_rate} Hz

Changes:
• RMS Change: {rms2_db - rms1_db:+.1f} dB
• Peak Change: {peak2_db - peak1_db:+.1f} dB
        """
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Analysis saved to: {save_path}")
        else:
            plt.show()
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze and compare two audio files')
    parser.add_argument('file1', help='First audio file (original)')
    parser.add_argument('file2', help='Second audio file (processed)')
    parser.add_argument('--output', '-o', help='Output image file (optional)')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                      help='Output format (default: png)')
    
    args = parser.parse_args()
    
    # Verify files exist
    for filepath in [args.file1, args.file2]:
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            sys.exit(1)
    
    # Create analyzer
    analyzer = AudioAnalyzer(args.file1, args.file2)
    
    # Generate output filename if not provided
    output_path = None
    if args.output:
        output_path = args.output
    else:
        base1 = os.path.splitext(os.path.basename(args.file1))[0]
        base2 = os.path.splitext(os.path.basename(args.file2))[0]
        output_path = f"audio_comparison_{base1}_vs_{base2}.{args.format}"
    
    # Create analysis
    print("Generating analysis...")
    analyzer.create_comparison_plot(save_path=output_path)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()