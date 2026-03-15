import qrcode
import os

# 关键：替换为你的有效Network URL
app_url = "http://10.7.28.77:8501"

# 高容错+大像素配置（手机易识别）
qr = qrcode.QRCode(
    version=3,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=15,  # 二维码像素放大，避免模糊
    border=4,
)
qr.add_data(app_url)
qr.make(fit=True)

# 保存到DataHack文件夹
img = qr.make_image(fill_color="black", back_color="white")
img_path = "streamlit_recycling_qr.png"
img.save(img_path)

print(f"✅ 二维码生成成功！")
print(f"📂 文件路径：{os.path.abspath(img_path)}")
print(f"🔗 访问链接：{app_url}")
