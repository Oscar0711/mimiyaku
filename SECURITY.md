# 安全性

## 回報安全性問題

如果你發現安全性問題，例如：

- 雲端服務的 API 金鑰可能外洩（被寫進記錄檔、設定檔，或送到不該送的地方）
- 處理影音檔或網址時，可能被利用來執行任意程式
- 下載模型時，可能被替換成惡意檔案

**請不要開公開的 Issue。** 請到 [Security → Report a vulnerability](https://github.com/Oscar0711/omni-translate/security/advisories/new) 私下回報，只有維護者看得到。

回報時請盡量附上：問題的說明、重現步驟、可能造成的影響。我會盡快回覆，修好後再公開說明。

## 金鑰怎麼保存

- 雲端語音辨識的金鑰存在 Windows 認證管理員（透過 `keyring`），不會寫進設定檔或記錄檔。
- 金鑰只會送到該服務自己的 API（阿里雲 DashScope、OpenAI、Google Cloud）。
- 回報問題時自動帶入的資訊只有版本、Windows 版本、顯示卡和模型名稱，不含檔案路徑和金鑰。
