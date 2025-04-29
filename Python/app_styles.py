# app_styles.py
class AppStyles:
    # Heading styles
    HEADING_FONT_SIZE = 18
    HEADING_FONT_WEIGHT = "bold"
    HEADING_STYLE = f"font-size: {HEADING_FONT_SIZE}px; font-weight: {HEADING_FONT_WEIGHT};"

    # Normal text styles
    TEXT_FONT_SIZE = 14
    TEXT_STYLE = f"font-size: {TEXT_FONT_SIZE}px;"

    # Button styles
    BUTTON_FONT_SIZE = 14
    BUTTON_STYLE = f"font-size: {BUTTON_FONT_SIZE}px;"

    # Input field styles
    INPUT_FONT_SIZE = 14
    INPUT_STYLE = f"font-size: {INPUT_FONT_SIZE}px;"

    # List item styles
    LIST_FONT_SIZE = 14
    LIST_STYLE = f"font-size: {LIST_FONT_SIZE}px;"

    # Note text styles (slightly smaller)
    NOTE_FONT_SIZE = 12
    NOTE_STYLE = f"font-size: {NOTE_FONT_SIZE}px; font-style: italic; color: gray;"

    # Graph title styles
    GRAPH_TITLE_FONT_SIZE = 16
    GRAPH_TITLE_STYLE = f"font-size: {GRAPH_TITLE_FONT_SIZE}px; font-weight: bold; color: white;"

    # Graph axis label styles
    GRAPH_LABEL_FONT_SIZE = 14
    GRAPH_LABEL_COLOR = "#FFFFFF"

    # Spacing between widgets
    WIDGET_SPACING = 25  # Pixels between widgets

    @staticmethod
    def get_group_box_style():
        return f"QGroupBox {{ {AppStyles.HEADING_STYLE} }}"

    @staticmethod
    def get_label_style():
        return f"QLabel {{ {AppStyles.TEXT_STYLE} }}"

    @staticmethod
    def get_button_style():
        return f"QPushButton {{ {AppStyles.BUTTON_STYLE} }}"

    @staticmethod
    def get_line_edit_style():
        return f"QLineEdit {{ {AppStyles.INPUT_STYLE} }}"

    @staticmethod
    def get_list_widget_style():
        return f"QListWidget {{ {AppStyles.LIST_STYLE} }}"

    @staticmethod
    def get_note_label_style():
        return f"QLabel {{ {AppStyles.NOTE_STYLE} }}"