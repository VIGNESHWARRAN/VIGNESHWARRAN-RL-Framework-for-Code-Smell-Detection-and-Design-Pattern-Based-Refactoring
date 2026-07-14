import javalang

def check_single_code(code_str: str) -> bool:
    try:
        try:
            javalang.parse.parse(code_str)
        except Exception:
            javalang.parse.parse(f"class _Dummy {{ {code_str} }}")
        return True
    except Exception:
        return False
