chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
    switch (request.action) {
        case "scanText":
            sendResponse({data: scanText()});
            break;
        case "scanImages":
            sendResponse({images: scanImages()});
            break;
        case "scanImage":
            scanImage(request.imgUrl);
            break;
    }

});

function scanText() {
    let textList = [];
    $("p, h1, h2, h3, h4, h5, h6, blockquote").each(function() {
        if ($(this).text().length < 80) {
            return
        }
        textList.push($(this).text());
        let fakeScore = Math.random();
        if (fakeScore < 0.2) {
            $(this).css(borderCSS("#A3DDCB"));
        } else if (fakeScore < 0.5) {
            $(this).css(borderCSS("#E8E9A1"));
        } else if (fakeScore < 0.8) {
            $(this).css(borderCSS("#E6B566"));
        } else {
            $(this).css(borderCSS("#E5707E"));
        }
    });
    console.log(textList);
    return textList;
}

function scanImages() {
    let images = [];
    $("img").each(function() {
        images.push($(this).attr("src"));
    });
    return images;
}

function scanImage(targetUrl) {
    // TODO Send image to server
}
function borderCSS(colour) {
    return {
        "box-shadow": `-4px 0px 1px -1px ${colour}`,
        "padding-left": "10px",
        "margin-left": "-10px"
    }
}
