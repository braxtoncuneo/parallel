import json
import sys
import os
import re

modes = [ "none", "web", "slide", "both" ]


slide_style  = "<style>\n"
#slide_style += ":root { --content-max-width: 90%; }\n"
slide_style += "img { width: 100%; height: 100%; object-fit: contain; }\n"
slide_style += ".segment_container { display: flex; flex-direction: row; }\n"
slide_style += ".segment { margin-left: 2%; margin-right: 2%; flex: 1; }\n"
slide_style += "</style>\n"


cell_start = "<div class=\"segment\" {cell_attr}>\n\n"
cell_end   = "</div>\n"

row_start = "<div class=\"segment_container\">\n"
row_start += cell_start

row_end = "</div>\n"
row_end = cell_end + row_end

slide_start  = "<div height: 100%>"
slide_start += row_start

slide_end   = "</div>"
slide_end   = row_end + slide_end



def slide_script(slide_count):
    global script_path
    script = open(script_path)
    code = script.read()
    script.close()
    code += f"slide_count = {slide_count}"
    code  = f"\n<script>\n{code}\n</script>\n"
    return code


def process_slide(content,slide_number):
    result  = f"<div id=\"slide_{slide_number}\" style=\"display: none;\"> "
    result += slide_start.format(cell_attr="style=\"flex: 1;\"")
    for line in content.splitlines():
        line = re.sub(r"{{footnote:[^}]*(}[^}]+)*}}","",line)
        directive = r"^[ \t]*<!--[ \t]*slider[ \t]*(?P<dir>[\w-]+)[ \t]*(?P<arg>[\w-]+)?[ \t]*-->[ \t]*$"
        match = re.match(directive,line)
        if match == None:
            result += line + '\n'
            continue
        d = match.group('dir')
        arg = match.group('arg')
        if arg == None:
            arg = 1
        if d == 'row-split' :
            result += row_end
            result += row_start.format(cell_attr=f"style=\"flex: {arg};\"")
        elif d == 'cell-split':
            result += cell_end
            result += cell_start.format(cell_attr=f"style=\"flex: {arg};\"")

    result += slide_end
    result +=  "</div>"
    return result


def filter_content(content,current_mode):
    web_content = "<div id=\"slide_web\">\n\n"
    slide_number = 0
    slides = [""]
    for line in content.splitlines():
            directive = r"^[ \t]*<!--[ \t]*slider[ \t]*(?P<dir>[\w-]+)[ \t]*-->[ \t]*$"
            match = re.match(directive,line)
            if match != None:
                d = match.group('dir')
                if d in modes:
                    index = modes.index(d)
                    current_mode = index
                elif d == 'split':
                     slides.append("")
                     slide_number += 1
            relpath = r"{{[ \t]*#relpath[ \t]*}}"
            revised_line = re.sub(relpath,".",line)
            if current_mode % 2  == 1 :
                if match == None:
                    web_content += revised_line + '\n'
            if current_mode // 2 == 1 :
                slides[slide_number] += revised_line + '\n'
    content = web_content + "</div>"
    for idx, slide in enumerate(slides):
        content += process_slide(slide,idx)
    content += "<script src=\"../slider_client.js\"></script>"
    content += slide_style
    return content, current_mode





def filter_chapter(chapter,current_mode=3):
    chapter['content'], _ = filter_content(chapter['content'],current_mode)
    for sub_item in chapter['sub_items']:
        if 'Chapter' not in sub_item:
              continue
        filter_chapter(sub_item['Chapter'])

script_path = None


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "supports": 
            sys.exit(0)

    context, book = json.load(sys.stdin)
    script_path = os.path.join(context['root'],'src')
    script_path = os.path.join(script_path,'slider_client.js')

    current_mode = 3

    for sect in book['sections']:
        if 'Chapter' not in sect:
              continue
        filter_chapter(sect['Chapter'])
    print(json.dumps(book))

