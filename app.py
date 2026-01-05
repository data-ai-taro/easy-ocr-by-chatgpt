import base64
import io
import os
from typing import Optional

import streamlit as st
from PIL import Image
from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"  # 画像入力対応モデル（必要なら変更）


def image_bytes_to_data_url(img_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def guess_mime(filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".webp"):
        return "image/webp"
    # jpg/jpeg fallback
    return "image/jpeg"


def normalize_image_bytes(uploaded_bytes: bytes) -> bytes:
    """
    画像が特殊形式/巨大すぎる場合の保険として、PILで読み→PNG/JPEGに再エンコードして安定化。
    """
    img = Image.open(io.BytesIO(uploaded_bytes))
    # 透過があるかどうかでPNG/JPEGを切り替え
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    out = io.BytesIO()
    if has_alpha:
        img = img.convert("RGBA")
        img.save(out, format="PNG", optimize=True)
    else:
        img = img.convert("RGB")
        img.save(out, format="JPEG", quality=95, optimize=True)
    return out.getvalue()


def ocr_with_openai(
    *,
    client: OpenAI,
    model: str,
    image_data_url: str,
    hint: Optional[str] = None,
) -> str:
    """
    画像から手書き文字をできるだけ忠実に抽出（日本語含む）
    """
    instruction = (
        "あなたは高精度なOCRです。与えられた画像内の手書き文字を読み取り、"
        "可能な限り正確にテキスト化してください。\n\n"
        "出力ルール:\n"
        "- 余計な説明はせず、抽出した本文のみを出力\n"
        "- 改行・段落・箇条書きは可能な範囲で維持\n"
        "- 読めない文字は「□」で置き換える（推測で断定しない）\n"
        "- 数字・記号・単位も可能な限り維持\n"
    )
    if hint:
        instruction += f"\n補足情報（任意）: {hint}\n"

    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {"type": "input_image", "image_url": image_data_url},
                ],
            }
        ],
        # OCRはブレが少ない方が嬉しいことが多いので低め
        temperature=0,
    )
    return (resp.output_text or "").strip()


def main():
    st.set_page_config(page_title="手書きOCR（ChatGPT API）", layout="centered")
    st.title("📝 手書きOCR（ChatGPT API / Responses）")
    st.caption("画像をアップロード → ChatGPT APIの画像入力で文字起こしします。")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.error("環境変数 OPENAI_API_KEY が設定されていません。")
        st.stop()

    client = OpenAI(api_key=api_key)

    with st.sidebar:
        st.subheader("設定")
        model = st.text_input("モデル", value=DEFAULT_MODEL)
        hint = st.text_area(
            "補足（任意）",
            placeholder="例：日本語、横書き、会議メモ、数字多め…など",
        )
        st.markdown("---")
        st.write("※ 個人情報・機密情報を含む画像の取り扱いは注意してください。")

    uploaded = st.file_uploader(
        "手書き文字の画像をアップロード（png/jpg/webp）",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if not uploaded:
        st.info("画像をアップロードするとOCRを実行できます。")
        return

    # 表示
    st.image(uploaded, caption=uploaded.name, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        do_normalize = st.checkbox("画像を再エンコードして安定化（推奨）", value=True)
    with col2:
        run = st.button("OCR実行", type="primary", use_container_width=True)

    if not run:
        return

    raw_bytes = uploaded.getvalue()
    mime = guess_mime(uploaded.name)

    try:
        img_bytes = normalize_image_bytes(raw_bytes) if do_normalize else raw_bytes
        # normalize後はmimeがpng/jpgに変わりうるので再判定
        if do_normalize:
            # normalize_image_bytesが透過あり→PNG、なし→JPEGなのでそれに合わせる
            mime = "image/png" if Image.open(io.BytesIO(img_bytes)).mode == "RGBA" else "image/jpeg"

        image_url = image_bytes_to_data_url(img_bytes, mime)

        with st.spinner("OCR中…"):
            text = ocr_with_openai(
                client=client,
                model=model,
                image_data_url=image_url,
                hint=hint.strip() or None,
            )

        st.subheader("抽出結果")
        st.text_area("OCRテキスト", value=text, height=300)

        st.download_button(
            "テキストをダウンロード（.txt）",
            data=text.encode("utf-8"),
            file_name="ocr_result.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except Exception as e:
        st.exception(e)


if __name__ == "__main__":
    main()
