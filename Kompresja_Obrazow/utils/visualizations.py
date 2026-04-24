import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim_metric


# ── metryki ──────────────────────────────────────────────────────────────────

def calculate_snr(original, reconstructed):
    og  = np.sum(original ** 2)
    dif = np.sum((original - reconstructed) ** 2)
    return 10 * np.log10(og / dif)

def calculate_psnr(original, reconstructed):
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255 ** 2 / mse)

def calculate_ssim(original, reconstructed):
    o = np.clip(original, 0, 255).astype(np.float64)
    r = np.clip(reconstructed, 0, 255).astype(np.float64)
    return ssim_metric(o, r, data_range=255.0)

def calculate_mse(original, reconstructed):
    return np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)


# ── pomocnicze FFT/DCT ────────────────────────────────────────────────────────

def _dct2(img):
    return dct(dct(img, axis=0, norm='ortho'), axis=1, norm='ortho')

def _fft2(img):
    return np.fft.fft2(img)


# ── wykresy istniejące ────────────────────────────────────────────────────────

def show_time_chart(times):
    labels = ['DCT', 'Scipy-DCT', 'Numpy-FFT', 'SFT']
    values = [np.sum(t) for t in times]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    plt.bar_label(bars, padding=3, fmt='%.4f')
    plt.ylabel('Czas (s)')
    plt.title('Czas pracy algorytmów')
    plt.tight_layout()
    plt.show()

def show_blocks_grid(split_image):
    plt.title("Obraz po podziale na bloki")
    for i in range(15):
        for j in range(15):
            plt.subplot(15, 15, i * 15 + j + 1)
            plt.imshow(split_image[i, j], cmap='gray')
            plt.axis('off')
    plt.show()

def show_decompression_efect(cropped_image, dct_image, fft_image, sft_image):
    plt.figure(figsize=(16, 5))
    for idx, (img, title) in enumerate([
        (cropped_image, "Oryginał"),
        (dct_image,     "DCT"),
        (fft_image,     "FFT"),
        (sft_image,     "SFT"),
    ]):
        plt.subplot(1, 4, idx + 1)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle("Efekt kompresji — porównanie wizualne")
    plt.tight_layout()
    plt.show()

def show_snr(cropped_image, dct_image, scipy_dct_image, fft_image, sft_image):
    results = [
        (dct_image,       "DCT"),
        (scipy_dct_image, "Scipy DCT"),
        (fft_image,       "Numpy FFT"),
        (sft_image,       "SFT"),
    ]
    plt.figure(figsize=(12, 10))
    for idx, (img, label) in enumerate(results):
        snr = calculate_snr(cropped_image, img)
        plt.subplot(2, 2, idx + 1)
        plt.title(f"{label}\nSNR = {snr:.2f} dB")
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# ── nowe metryki — główny wykres Remka ────────────────────────────────────────

def show_metrics_comparison(cropped_image, dct_image, scipy_dct_image, fft_image, sft_image):
    """
    Wykres zbiorczy metryk dla wszystkich czterech algorytmów.
    Panele: SNR, PSNR, SSIM (bar chart) + MSE (osobny panel, skala log).
    """
    algorithms = ['DCT', 'Scipy\nDCT', 'FFT', 'SFT']
    images     = [dct_image, scipy_dct_image, fft_image, sft_image]
    colors     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    snr_vals  = [calculate_snr(cropped_image, img)  for img in images]
    psnr_vals = [calculate_psnr(cropped_image, img) for img in images]
    ssim_vals = [calculate_ssim(cropped_image, img) for img in images]
    mse_vals  = [calculate_mse(cropped_image, img)  for img in images]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Porównanie metryk jakości — wszystkie algorytmy", fontsize=13, fontweight='bold')

    def _bar(ax, vals, title, ylabel, fmt='.2f', highlight_max=True):
        bars = ax.bar(algorithms, vals, color=colors, edgecolor='white', linewidth=0.5)
        best = np.argmax(vals) if highlight_max else np.argmin(vals)
        bars[best].set_edgecolor('black')
        bars[best].set_linewidth(2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=9)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    _bar(axes[0, 0], snr_vals,  "SNR (Signal-to-Noise Ratio)",           "dB",  highlight_max=True)
    _bar(axes[0, 1], psnr_vals, "PSNR (Peak SNR)",                        "dB",  highlight_max=True)
    _bar(axes[1, 0], ssim_vals, "SSIM (Structural Similarity Index)",     "",    highlight_max=True)

    # MSE — skala log, bo wartości mogą się bardzo różnić
    bars = axes[1, 1].bar(algorithms, mse_vals, color=colors, edgecolor='white', linewidth=0.5)
    best_mse = np.argmin(mse_vals)
    bars[best_mse].set_edgecolor('black')
    bars[best_mse].set_linewidth(2)
    for bar, v in zip(bars, mse_vals):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                        f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].set_title("MSE (Mean Squared Error)", fontweight='bold')
    axes[1, 1].set_ylabel("Błąd kwadratowy (↓ lepiej)")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()


# ── błąd aproksymacji — DCT, FFT, SFT ────────────────────────────────────────

def show_correlation(cropped_image, sft_keep_fraction=0.01):
    """
    Krzywa błędu aproksymacji: jak maleje błąd energii
    w zależności od liczby zachowanych współczynników.
    DCT i FFT jako ciągłe krzywe, SFT jako punkt (działa przy stałym keep_fraction).
    """
    FC = _dct2(cropped_image)
    FF = _fft2(cropped_image)

    sorted_dct = np.sort(np.abs(FC).flatten())[::-1]
    sorted_fft = np.sort(np.abs(FF).flatten())[::-1]
    total_dct  = np.sum(FC ** 2)
    total_fft  = np.sum(np.abs(FF) ** 2)

    M_max = min(len(sorted_dct), 50000)
    errors_dct, errors_fft = [], []
    for m in range(1, M_max):
        errors_dct.append((total_dct - np.sum(sorted_dct[:m] ** 2)) / total_dct)
        errors_fft.append((total_fft - np.sum(sorted_fft[:m] ** 2)) / total_fft)

    # Punkt SFT — ile współczynników faktycznie zachowuje przy keep_fraction
    H, W  = cropped_image.shape
    k_sft = max(int(H * W * sft_keep_fraction), 1)
    # błąd SFT = błąd FFT przy tych samych k najsilniejszych współczynnikach
    sft_error = (total_fft - np.sum(sorted_fft[:k_sft] ** 2)) / total_fft
    sft_error = max(sft_error, 1e-15)

    plt.figure(figsize=(10, 5))
    plt.plot(np.log10(errors_dct), color='red',  linewidth=1.5, label='DCT')
    plt.plot(np.log10(errors_fft), color='blue', linewidth=1.5, label='FFT')
    plt.scatter([k_sft], [np.log10(sft_error)], color='green', s=80, zorder=5,
                label=f'SFT (k={k_sft}, {sft_keep_fraction*100:.1f}%)')
    plt.axvline(k_sft, color='green', linestyle=':', linewidth=1, alpha=0.6)
    plt.xlabel("M — liczba zachowanych współczynników")
    plt.ylabel("log₁₀(błąd energii)")
    plt.title("Błąd aproksymacji — DCT vs FFT vs SFT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── widmo amplitudy 2D — DCT, FFT, SFT ───────────────────────────────────────

def show_coeffcients(cropped_image, sft_keep_fraction=0.01):
    """
    Widmo amplitudowe (log|F|) dla wszystkich trzech metod.
    SFT pokazuje sparse spectrum — widać które składowe zostały wybrane.
    """
    FC = _dct2(cropped_image)
    FF = _fft2(cropped_image)
    FF_shifted = np.fft.fftshift(FF)

    # Sparse spectrum SFT (top-k z FFT, przybliżenie efektu SFT)
    H, W  = cropped_image.shape
    k_sft = max(int(H * W * sft_keep_fraction), 1)
    flat  = np.abs(FF).flatten()
    thresh = np.sort(flat)[::-1][k_sft - 1]
    FF_sparse = np.where(np.abs(FF) >= thresh, FF, 0)
    FF_sparse_shifted = np.fft.fftshift(FF_sparse)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Widmo amplitudowe — log|F| dla każdego algorytmu", fontsize=12)

    axes[0].imshow(np.log(1e-5 + np.abs(FC)), cmap='inferno')
    axes[0].set_title("DCT — energia skupiona w rogu (lewy górny)")
    axes[0].axis('off')

    axes[1].imshow(np.log(1e-5 + np.abs(FF_shifted)), cmap='inferno')
    axes[1].set_title("FFT — energia skupiona w centrum")
    axes[1].axis('off')

    axes[2].imshow(np.log(1e-5 + np.abs(FF_sparse_shifted)), cmap='inferno')
    axes[2].set_title(f"SFT — sparse spectrum (top {sft_keep_fraction*100:.1f}% = {k_sft} wsp.)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()


# ── faza — FFT, DCT, SFT ─────────────────────────────────────────────────────

def show_phase_comparison(cropped_image, threshold=50, sft_keep_fraction=0.01):
    """
    Porównanie widma fazowego (przekrój 1D) dla FFT, DCT i SFT.
    """
    FF = _fft2(cropped_image)
    FC = _dct2(cropped_image)

    ff_row   = np.fft.fftshift(FF[0, :])
    freqs    = np.fft.fftshift(np.fft.fftfreq(ff_row.size))

    # FFT faza
    mask_fft   = np.abs(ff_row) >= threshold
    phase_fft  = np.angle(ff_row)
    kept_fft   = 100 * np.sum(mask_fft) / mask_fft.size

    # DCT — współczynniki są rzeczywiste, faza = 0 (dodatnie) lub π (ujemne)
    fc_row     = FC[0, :]
    fc_x       = np.linspace(-0.5, 0.5, len(fc_row))
    phase_dct  = np.angle(fc_row.astype(complex))   # 0 lub ±π
    fc_abs     = np.abs(fc_row)
    dct_thresh = np.percentile(fc_abs, 75)           # zachowaj top 25%
    mask_dct   = fc_abs >= dct_thresh
    kept_dct   = 100 * np.sum(mask_dct) / mask_dct.size

    # SFT — top-k współczynniki FFT (przekrój wiersza)
    H, W   = cropped_image.shape
    k_sft  = max(int(ff_row.size * sft_keep_fraction), 1)
    thresh_sft = np.sort(np.abs(ff_row))[::-1][k_sft - 1]
    mask_sft   = np.abs(ff_row) >= thresh_sft
    kept_sft   = 100 * np.sum(mask_sft) / mask_sft.size

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))
    fig.suptitle("Faza — porównanie wszystkich algorytmów (przekrój 1D)", fontsize=12, fontweight='bold')

    # Panel 1: FFT
    axes[0].plot(freqs, phase_fft, color='steelblue', lw=0.8, alpha=0.25, label='przed kompresją')
    kept_line = np.where(mask_fft, phase_fft, np.nan)
    axes[0].plot(freqs, kept_line, color='steelblue', lw=1.4,
                 label=f'FFT zachowane ({kept_fft:.1f}%, próg amp={threshold})')
    axes[0].set_title("FFT")
    axes[0].set_ylabel("Faza (rad)")
    axes[0].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[0].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Panel 2: DCT (faza 0/π — bo wartości rzeczywiste)
    axes[1].plot(fc_x, phase_dct, color='tomato', lw=0.8, alpha=0.3, label='wszystkie wsp.')
    dct_kept_line = np.where(mask_dct, phase_dct, np.nan)
    axes[1].plot(fc_x, dct_kept_line, color='tomato', lw=1.4,
                 label=f'DCT zachowane ({kept_dct:.1f}%, top 25% amplitudy)')
    axes[1].set_title("DCT — faza binarna (0 = dodatni wsp., π = ujemny wsp.)")
    axes[1].set_ylabel("Faza (rad)")
    axes[1].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[1].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[1].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Panel 3: SFT (sparse FFT — te same biny co FFT ale wybrane losowo/adaptacyjnie)
    axes[2].plot(freqs, phase_fft, color='seagreen', lw=0.8, alpha=0.25, label='FFT pełny (tło)')
    sft_kept_line = np.where(mask_sft, phase_fft, np.nan)
    axes[2].plot(freqs, sft_kept_line, color='seagreen', lw=1.4,
                 label=f'SFT zachowane ({kept_sft:.1f}%, top {sft_keep_fraction*100:.1f}%)')
    axes[2].set_title("SFT — zachowane sparse współczynniki FFT")
    axes[2].set_ylabel("Faza (rad)")
    axes[2].set_xlabel("Częstotliwość")
    axes[2].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[2].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axes[2].set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()