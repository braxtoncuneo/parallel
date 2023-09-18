
var slide_number = 0;
var slide_mode   = false;
var slide_count  = 0;


while (true) {
    slide = document.getElementById("slide_"+  slide_count);
    if (slide === null) {
        break;
    }
    slide_count = slide_count + 1;
}

function slide_handler(delta) {

    if(slide_mode == false){
        return;
    }

    var new_number = slide_number + delta;
    if(new_number >= slide_count){
        new_number = new_number % slide_count;
    } else if (new_number < 0) {
        new_number = 0;
    }

    old_slide = document.getElementById("slide_"+slide_number);
    new_slide = document.getElementById("slide_"+  new_number);
    old_slide.style.display = 'none'
    new_slide.style.display = 'block'
    
    slide_number = new_number;

}

content_default_max = document.documentElement.style.getPropertyValue('--content-max-width')

function mode_flip() {
    slide_mode = ! slide_mode;
    slide = document.getElementById("slide_"+slide_number);
    web   = document.getElementById("slide_web");
    content = document.getElementById("content");
    if (slide_mode) {
        web.style.display = 'none'
        slide.style.display = 'block'
        document.documentElement.style.setProperty('--content-max-width', '90%')
        content.style.setProperty('font-size', '1vw')
    } else {
        web.style.display = 'block'
        slide.style.display = 'none'
        document.documentElement.style.setProperty('--content-max-width', content_default_max)
        content.style.removeProperty('font-size')
    }

    main = document.getElementsByTagName('main')[0];
    nodes = main.children
    for(i=nodes.length-1; i>=0; i--){
        node = nodes[i]
        if( (node.tagName == 'P') || (node.tagName == 'HR') ){
            if( slide_mode ){
                node.style.display = 'none'
            } else{
                node.style.removeProperty('display')
            }
        } else {
            break;
        }
    }
}


document.addEventListener('keydown',function(event){

    if (event.code == "Escape") {
        mode_flip()
    }

    if (!slide_mode){
        return;
    }

    if(event.code == "Minus" ){
        slide_handler(-1);
    } else if (event.code == "Equal") {
        slide_handler(1);
    }
    
});


