const canvas = new fabric.Canvas('c');

fabric.Image.fromURL(imgSrc, function(img) {
    canvas.setWidth(img.width);
    canvas.setHeight(img.height);
    canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
});

// DRAW
let rect, isDown;

canvas.on('mouse:down', function(o) {
    isDown = true;
    let p = canvas.getPointer(o.e);

    rect = new fabric.Rect({
        left: p.x,
        top: p.y,
        width: 0,
        height: 0,
        stroke: 'red',
        fill: 'transparent'
    });

    canvas.add(rect);
});

canvas.on('mouse:move', function(o) {
    if (!isDown) return;
    let p = canvas.getPointer(o.e);

    rect.set({
        width: p.x - rect.left,
        height: p.y - rect.top
    });

    canvas.renderAll();
});

canvas.on('mouse:up', () => isDown = false);

// LOAD
fetch(`/load/${imageId}`)
.then(r => r.json())
.then(data => {
    data.forEach(b => {
        canvas.add(new fabric.Rect({
            left: b.x,
            top: b.y,
            width: b.w,
            height: b.h,
            stroke: 'green',
            fill: 'transparent'
        }));
    });
});

// SAVE
function saveBoxes(){
    const boxes = canvas.getObjects().map(o => ({
        x:o.left,
        y:o.top,
        w:o.width * o.scaleX,
        h:o.height * o.scaleY,
        label:0
    }));

    fetch(`/save/${imageId}`, {
        method:"POST",
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify(boxes)
    });
}

// AUTO DETECT
function autoDetect(){
    fetch(`/detect/${imageId}`)
    .then(r => r.json())
    .then(data => {
        data.forEach(b => {
            canvas.add(new fabric.Rect({
                left: b.x,
                top: b.y,
                width: b.w,
                height: b.h,
                stroke: 'blue',
                fill: 'transparent'
            }));
        });
    });
}