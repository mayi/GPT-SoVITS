from st_pages import Page, show_pages, add_page_title

add_page_title("Streamlit WebUI")

show_pages([
    Page("streamlit_pages/uvr5.py", "伴奏人声分离&去混响&去回声", ":musical_note:"),
    Page("streamlit_pages/subfix.py", "音频文字校准", ":memo:"),
])