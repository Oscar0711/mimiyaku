# 參與開發

謝謝你願意幫忙！不管是回報問題、提出建議，還是送程式碼，都很歡迎。

## 回報問題、提出建議

- **程式出錯、結果明顯不對**：開一則 [Issue](https://github.com/Oscar0711/omni-translate/issues/new/choose)，選「回報問題」。從程式的「設定 → 其他 → 回報問題」開啟的話，版本和顯示卡資訊會自動填好。
- **想要新功能**：開一則 Issue，選「功能建議」。
- **使用問題、還不成熟的想法、心得分享**：到 [討論區](https://github.com/Oscar0711/omni-translate/discussions)。
- **安全性問題**：請看 [SECURITY.md](SECURITY.md)，不要公開回報。

## 開發環境

需要 Windows 10/11、[uv](https://docs.astral.sh/uv/)、ffmpeg。要跑本地模型的話，需要 NVIDIA 顯示卡，驅動程式要支援 CUDA 13。

```bash
git clone https://github.com/Oscar0711/omni-translate.git
cd omni-translate
uv sync
uv run pytest            # 單元測試，不需要顯示卡和模型
uv run omni2-gui         # 開啟介面
```

架構、各階段的設計理由和實測數據都寫在 [docs/design.md](docs/design.md)，改動前建議先看相關段落。

## 送出修改（Pull Request）

1. 比較大的改動，請先開 Issue 或討論，確認方向後再動手，避免白做。
2. 從 `main` 開一個新分支，一個 PR 只做一件事。
3. 新功能或修正請附上測試（`tests/`）；送出前確認 `uv run pytest` 全部通過。
4. 程式風格跟周圍的程式一致：註解和介面文字用繁體中文，說明「為什麼」而不是「做了什麼」。
5. 動到辨識、翻譯、講者分離的效果時，請附上實測前後的比較（`tools/bench_*.py`）。

## 授權

本專案以 [GNU AGPL-3.0](LICENSE) 授權。送出 Pull Request，即表示你同意你的貢獻以相同授權釋出。
