# SC AI — RunPod Serverless GPU Lyrics Auto-Sync Worker

Dedicated GPU worker for high-accuracy singing lyrics forced alignment (**HTDemucs Vocal Isolation + Stable-TS Whisper `large-v3-turbo` + Silero VAD**).

## Cara Deploy ke RunPod Serverless (Opsi Hemat 16 GB GPU)

1. Push isi folder `runpod-lyrics-sync/` ini ke repository GitHub baru (misal `scai-lyrics-sync`).
2. Di **RunPod Dashboard → Serverless → New Endpoint → Import Git Repository**:
   - Pilih repo GitHub `scai-lyrics-sync` (branch `main`, Dockerfile path `Dockerfile`).
   - **GPU**: pilih kelas hemat **16 GB (RTX A4000 / 4000 Ada)** atau **24 GB (L4 / RTX 3090 / A5000)**.
   - **Active Workers**: `0` (Flex Worker — biaya `$0` saat idle).
   - **Max Workers**: `3` (sesuai kebutuhan).
   - **Idle Timeout**: `5` detik.
   - **Execution Timeout**: `180` detik.
   - **Container Disk**: `15 GB`.
3. Setelah status endpoint **Ready**, salin **Endpoint ID** lalu tempel di **SC AI Admin Panel → Pengaturan API → GPU Auto-Sync Lirik (RunPod Serverless)**.
