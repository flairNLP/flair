TAGGED_ENTITY = '''
<mark class="entity" style="background: {color}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 3; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
    {entity}
    <span style="font-size: 0.8em; font-weight: bold; line-height: 3; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">{label}</span>
</mark>
'''

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>Flair</title>
    </head>

    <body style="font-size: 16px; font-family: 'Segoe UI'; padding: 4rem 2rem">{text}</body>
</html>
"""
