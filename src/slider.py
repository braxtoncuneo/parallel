import os
import re

cwd = os.getcwd()

src_path = os.path.join(cwd,"src")

modes = [ "none", "web", "slide", "both" ]


path_list = {}


def generate_slide(slide_content,slide_number,path,file):
    slide_name = f"{slide_number}.md"
    file_relpath = os.path.realpath(path,src_path)

    seq = []
    while True:
        head, tail = os.path.split(file_relpath)

        if tail != "":
            seq.append(tail)
        elif head != "":
            seq.append(head)
            break
    seq.reverse            

    base = path_list
    for p in seq:
        if p not in base:
            base[p] = {}
        base = base[p]
    base[file] = file

    file_relpath = os.path.join(file_relpath,file)

    slide_path = os.path.join(file_relpath,slide_name)
    slide_file = open(slide_path,mode='w')
    slide_file.write(slide_content)

def process_file(subdir,file):
    path = os.path.join(subdir,file)

    slide_number = 0

    file_in = open(path)
    web_content   = ""
    slide_content = ""
    for idx, line in enumerate(file_in):
        directive = r"^[ \t]*<!--[ \t]*slider[ \t]*(?P<dir>\w+)[ \t]*-->[ \t]*$"
        match_list = re.match(directive)
        for match in match_list:
            d = match.group('dir')
            for i in range(4):
                if d != modes[i]:
                    continue
                if d % 2 == 1:
                    web_content += line
                if d / 2 == 1:
                    slide_content += line
            if (d == "split") or (idx):
                generate_slide(slide_number,path,file)
                slide_content  = ""
                slide_number  += 1
    
    generate_slide(slide_number,path,file)
    slide_content  = ""
    slide_number  += 1

    file_in.close()


for subdir, dirs, files in os.walk(cwd):
    for file in files:
        if not file.endswith(".md"):
            continue

        process_file(subdir,file)

        


