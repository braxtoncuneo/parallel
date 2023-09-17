import json
import sys
import re

modes = [ "none", "web", "slide", "both" ]



def filter_content(content,current_mode):
    result = ""
    for line in content.splitlines():
            directive = r"^[ \t]*<!--[ \t]*slider[ \t]*(?P<dir>\w+)[ \t]*-->[ \t]*$"
            match = re.match(directive,line)
            if match != None:
                d = match.group('dir')
                if d in modes:
                    index = modes.index(d)
                    current_mode = index
            relpath = r"{{[ \t]*#relpath[ \t]*}}"
            revised_line = re.sub(relpath,".",line)
            if revised_line != line:
                print(revised_line,file=sys.stderr)
            if current_mode % 2 == 1:
                result += revised_line + '\n'
    return result, current_mode


def filter_chapter(chapter,current_mode=3):
    chapter['content'], _ = filter_content(chapter['content'],current_mode)
    for sub_item in chapter['sub_items']:
        if 'Chapter' not in sub_item:
              continue
        filter_chapter(sub_item['Chapter'])
         

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == "supports": 
            sys.exit(0)

    context, book = json.load(sys.stdin)

    current_mode = 3

    for sect in book['sections']:
        if 'Chapter' not in sect:
              continue
        filter_chapter(sect['Chapter'])
    print(json.dumps(book))

