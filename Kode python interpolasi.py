import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random

# ==========================================
# TAHAP 1: PERSIAPAN DATA (ALGORITMA 5.1)
# ==========================================
def persiapan_data(x_penuh, y_penuh, n_validasi, seed_acak):
    if np.issubdtype(x_penuh.dtype, np.number):
        mask_x = ~np.isnan(x_penuh)
    else:
        mask_x = np.ones(len(x_penuh), dtype=bool)
        
    y_numerik = pd.to_numeric(y_penuh, errors='coerce')
    mask_y = ~np.isnan(y_numerik)
    
    mask = mask_x & mask_y
    x_bersih = x_penuh[mask]
    y_bersih = y_numerik[mask]
    
    N = len(x_bersih)
    if N <= n_validasi + 2:
        raise ValueError("Data valid terlalu sedikit untuk menyisihkan n_validasi.")

    random.seed(seed_acak)
    indeks_validasi = random.sample(range(1, N - 1), n_validasi)
    indeks_validasi.sort()

    mask_validasi = np.zeros(N, dtype=bool)
    mask_validasi[indeks_validasi] = True
    
    x_validasi = x_bersih[mask_validasi]
    y_validasi = y_bersih[mask_validasi]
    
    x_basis = x_bersih[~mask_validasi]
    y_basis = y_bersih[~mask_validasi]
    
    print(f"Data dipisah: {len(x_basis)} titik basis, {len(x_validasi)} titik validasi.")
    return x_basis, y_basis, x_validasi, y_validasi

# ==========================================
# TAHAP 2: ALGORITMA INTERPOLASI
# ==========================================

# --- A. Interpolasi Linier (Algoritma 5.2) ---
def interpolasi_linier(x_basis, y_basis, x_query):
    for i in range(len(x_basis) - 1):
        if x_basis[i] <= x_query <= x_basis[i+1]:
            slope = (y_basis[i+1] - y_basis[i]) / (x_basis[i+1] - x_basis[i])
            return y_basis[i] + slope * (x_query - x_basis[i])
    return np.nan 

def evaluasi_linier(x_basis, y_basis, x_query_array):
    return np.array([interpolasi_linier(x_basis, y_basis, xq) for xq in x_query_array])

# --- B. Interpolasi Newton (Algoritma 5.3a & 5.3b) ---
def hitung_koefisien_newton(x_basis, y_basis):
    N = len(x_basis)
    tabel = np.zeros((N, N))
    tabel[:, 0] = y_basis 
    
    for j in range(1, N):
        for i in range(N - j):
            tabel[i][j] = (tabel[i+1][j-1] - tabel[i][j-1]) / (x_basis[i+j] - x_basis[i])
    
    koefisien = tabel[0, :] 
    return koefisien

def evaluasi_newton(koefisien, x_basis, x_query_array):
    N = len(koefisien)
    hasil_array = []
    
    for x_query in x_query_array:
        hasil = koefisien[N-1]
        for k in range(N-2, -1, -1):
            hasil = max_val = hasil * (x_query - x_basis[k]) + koefisien[k]
        hasil_array.append(hasil)
        
    return np.array(hasil_array)

# --- C. Interpolasi Lagrange (Algoritma 5.4) ---
def evaluasi_lagrange(x_basis, y_basis, x_query_array):
    N = len(x_basis)
    hasil_array = []
    
    for x_query in x_query_array:
        hasil = 0
        for i in range(N):
            Li = 1.0
            for j in range(N):
                if i != j:
                    Li = Li * (x_query - x_basis[j]) / (x_basis[i] - x_basis[j])
            hasil += Li * y_basis[i]
        hasil_array.append(hasil)
        
    return np.array(hasil_array)

# ==========================================
# TAHAP 3: EVALUASI ERROR (ALGORITMA 5.5)
# ==========================================
def hitung_error(y_acuan, y_estimasi):
    y_acuan = np.array(y_acuan)
    y_estimasi = np.array(y_estimasi)
    Et = np.abs((y_acuan - y_estimasi) / y_acuan) * 100
    rmse = np.sqrt(np.mean((y_acuan - y_estimasi)**2))
    return Et, rmse

# SCRIPT TAMBAHAN: Menampilkan Error di Terminal secara Terstruktur
def cetak_tabel_error(x_validasi, y_validasi, y_val_lin, y_val_newt, y_val_lagr, nama_kasus):
    """
    Fungsi khusus untuk mencetak rangkuman parameter error ke terminal.
    Menampilkan Error Relatif (Et %) per titik dan nilai RMSE global metode.
    """
    et_lin, rmse_lin = hitung_error(y_validasi, y_val_lin)
    et_newt, rmse_newt = hitung_error(y_validasi, y_val_newt)
    et_lagr, rmse_lagr = hitung_error(y_validasi, y_val_lagr)
    
    print("\n" + "="*75)
    print(f" TABEL RINGKASAN EVALUASI ERROR - {nama_kasus.upper()} ")
    print("="*75)
    print(f"{'Titik (X)':<12} | {'Nilai Asli':<12} | {'Et% Linier':<12} | {'Et% Newton':<12} | {'Et% Lagrange':<12}")
    print("-"*75)
    
    # Cetak Error Relatif (Et) untuk tiap titik validasi yang disisihkan
    for i in range(len(x_validasi)):
        print(f"{x_validasi[i]:<12.2f} | {y_validasi[i]:<12.4f} | {et_lin[i]:<11.4f}% | {et_newt[i]:<11.4f}% | {et_lagr[i]:<11.4f}%")
    
    print("-"*75)
    # Cetak nilai Root Mean Square Error (RMSE) kumulatif di bawah tabel
    print(f"{'RMSE METODE KESELURUHAN':<27} | {rmse_lin:<12.5f} | {rmse_newt:<12.5f} | {rmse_lagr:<12.5f}")
    print("="*75 + "\n")

# ==========================================
# TAHAP 4: VISUALISASI (ALGORITMA 5.6)
# ==========================================
def buat_visualisasi(x_basis, y_basis, x_validasi, y_validasi, 
                     koef_newton, y_val_lin, y_val_newt, y_val_lagr, judul, xlabel, ylabel, nama_file):
    
    x_halus = np.linspace(min(x_basis), max(x_basis), 200)
    y_lin_halus = evaluasi_linier(x_basis, y_basis, x_halus)
    y_newton_halus = evaluasi_newton(koef_newton, x_basis, x_halus)
    y_lagrange_halus = evaluasi_lagrange(x_basis, y_basis, x_halus)
    
    _, rmse_lin = hitung_error(y_validasi, y_val_lin)
    _, rmse_newt = hitung_error(y_validasi, y_val_newt)
    _, rmse_lagr = hitung_error(y_validasi, y_val_lagr)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(x_halus, y_lin_halus, label=f'Linier (RMSE: {rmse_lin:.3f})', color='blue', linestyle='--')
    ax1.plot(x_halus, y_newton_halus, label=f'Newton (RMSE: {rmse_newt:.3f})', color='green', alpha=0.7)
    ax1.plot(x_halus, y_lagrange_halus, label=f'Lagrange (RMSE: {rmse_lagr:.3f})', color='orange', alpha=0.7)
    
    ax1.scatter(x_basis, y_basis, color='black', label='Titik Basis (Data)', zorder=5)
    ax1.scatter(x_validasi, y_validasi, color='red', marker='*', s=150, label='Titik Validasi (Acuan)', zorder=6)
    
    ax1.set_title(f"Perbandingan Metode Interpolasi\n{judul}")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    indeks = np.arange(len(x_validasi))
    lebar = 0.25
    
    res_lin = y_validasi - y_val_lin
    res_newt = y_validasi - y_val_newt
    res_lagr = y_validasi - y_val_lagr
    
    ax2.bar(indeks - lebar, res_lin, lebar, label='Residual Linier', color='blue', alpha=0.7)
    ax2.bar(indeks, res_newt, lebar, label='Residual Newton', color='green', alpha=0.7)
    ax2.bar(indeks + lebar, res_lagr, lebar, label='Residual Lagrange', color='orange', alpha=0.7)
    
    ax2.axhline(0, color='black', linewidth=1)
    ax2.set_title(f"Residual Titik Validasi (Acuan - Estimasi)")
    ax2.set_xticks(indeks)
    ax2.set_xticklabels([f"x={x:.1f}" for x in x_validasi])
    ax2.set_ylabel("Nilai Residual")
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(nama_file, dpi=150)
    print(f"Grafik berhasil disimpan sebagai: {nama_file}")
    plt.show()

# ==========================================
# MAIN SCRIPT
# ==========================================
if __name__ == "__main__":
    
    print("--- MEMULAI KASUS A: TIME SERIES (CSV) ---")
    try:
        df_a = pd.read_csv('datasst_2020.csv')
        if not np.issubdtype(df_a['time'].dtype, np.number):
            x_a = np.arange(1, len(df_a) + 1)
        else:
            x_a = df_a['time'].values
        y_a = df_a['sst'].values
        
        xa_bas, ya_bas, xa_val, ya_val = persiapan_data(x_a, y_a, n_validasi=3, seed_acak=2026)
        koef_newton_a = hitung_koefisien_newton(xa_bas, ya_bas)
        
        # Hitung estimasi titik validasi Kasus A
        ya_val_lin = evaluasi_linier(xa_bas, ya_bas, xa_val)
        ya_val_newt = evaluasi_newton(koef_newton_a, xa_bas, xa_val)
        ya_val_lagr = evaluasi_lagrange(xa_bas, ya_bas, xa_val)
        
        # CETAK KE TERMINAL
        cetak_tabel_error(xa_val, ya_val, ya_val_lin, ya_val_newt, ya_val_lagr, "Kasus A (Time Series CSV)")
        
        # BUAT GRAFIK
        buat_visualisasi(xa_bas, ya_bas, xa_val, ya_val, koef_newton_a, 
                         ya_val_lin, ya_val_newt, ya_val_lagr,
                         "Kasus A: Time Series SST 2020", "Bulan ke-", "SST (°C)", "Plot_KasusA.png")
    except Exception as e:
        print(f"Gagal memproses Kasus A: {e}")

    print("\n--- MEMULAI KASUS B: DATA SPASIAL (NETCDF) ---")
    try:
        ds = xr.open_dataset('datasst_2020.nc')
        nama_var = 'sst' if 'sst' in ds.variables else list(ds.data_vars)[0]
        
        mid_lat_idx = len(ds['lat']) // 2 
        data_1d = ds[nama_var].isel(lat=mid_lat_idx) 
        
        x_b = data_1d['lon'].values
        y_b = data_1d.values
        
        xb_bas, yb_bas, xb_val, yb_val = persiapan_data(x_b, y_b, n_validasi=4, seed_acak=100)
        
        urut = np.argsort(xb_bas)
        xb_bas = xb_bas[urut]
        yb_bas = yb_bas[urut]
        
        koef_newton_b = hitung_koefisien_newton(xb_bas, yb_bas)
        
        # Hitung estimasi titik validasi Kasus B
        yb_val_lin = evaluasi_linier(xb_bas, yb_bas, xb_val)
        yb_val_newt = evaluasi_newton(koef_newton_b, xb_bas, xb_val)
        yb_val_lagr = evaluasi_lagrange(xb_bas, yb_bas, xb_val)
        
        # CETAK KE TERMINAL
        cetak_tabel_error(xb_val, yb_val, yb_val_lin, yb_val_newt, yb_val_lagr, "Kasus B (Spasial NetCDF)")
        
        # BUAT GRAFIK
        buat_visualisasi(xb_bas, yb_bas, xb_val, yb_val, koef_newton_b, 
                         yb_val_lin, yb_val_newt, yb_val_lagr,
                         f"Kasus B: Spasial SST (Lat: {ds['lat'].values[mid_lat_idx]:.2f})", 
                         "Longitude (°)", "SST (°C)", "Plot_KasusB.png")
    except Exception as e:
        print(f"Gagal memproses Kasus B: {e}")