chrome.runtime.onMessage.addListener(async function(request, sender, sendResponse) {
    switch (request.action) {
        case "scanText":
            const textData = await scanText();
            sendResponse(textData);
            break;
        case "scrapeImages":
            sendResponse({images: scrapeImages()});
            break;
        case "scanImages":
            const imagesData = await scanImages();
            sendResponse(imagesData);
            break;
        case "scanImage":
            const scanData = await scanImage(request.imgUrl);
            sendResponse({data: scanData});
            break;
    }
});

async function scanText() {
    let textList = [];
    let elements = [];
    let scores = [];

    $("p, h1, h2, h3, h4, h5, h6, blockquote").each(function() {
        if ($(this).text().length < 80) {
            return;
        }
        textList.push($(this).text());
        elements.push($(this));  // Store the jQuery element to style it later
    });

    // Send the collected text to the server
    try {
        const response = await fetch('https://example.com/scanText', { // Replace with actual server URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ texts: textList })
        });

        const data = await response.json();
        scores = data.scores;

        // Assume `data` contains an array of scores corresponding to the text blocks
        data.scores.forEach((fakeScore, index) => {
            if (fakeScore < 0.2) {
                elements[index].css(borderCSS("#A3DDCB"));
            } else if (fakeScore < 0.5) {
                elements[index].css(borderCSS("#E8E9A1"));
            } else if (fakeScore < 0.8) {
                elements[index].css(borderCSS("#E6B566"));
            } else {
                elements[index].css(borderCSS("#E5707E"));
            }
        });

        return { status: "OK", text: textList, scores: data.scores }; // Return the response from the server

    } catch (error) {
        console.error("Error sending text to the server:", error);
        return { status: "Failed to send text to the server" };
    }
}

function scrapeImages() {
    let images = [];
    $("img").each(function() {
        images.push($(this).attr("src"));
    });
    return images;
}

async function scanImages() {
    const images = scrapeImages();
    const scores = [];
    for (let img in images) {
        const imgFakeScore = await scanImage(img);
        if (imgFakeScore.status !== "OK") {
            return { status: "Failed to scan an image" };
        }
        scores.push(imgFakeScore);
    }
    return { status: "OK", scores: scores, images: images };
}

// Modified scanImage to send image URL to the server and get a fake score
async function scanImage(targetUrl) {
    try {
        const response = await fetch('https://example.com/scanImage', {  // Replace with actual server URL
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ imageUrl: targetUrl })
        });

        const data = await response.json();

        // TODO: Change this to the actual key used in the response
        const fakeScore = Math.random();
        console.log("Image scan result:", fakeScore);
        return { status: "OK", score: fakeScore };

    } catch (error) {
        console.error("Error sending image to the server:", error);
        return { status: "Failed to send image to the server" };
    }
}

function borderCSS(colour) {
    return {
        "box-shadow": `-4px 0px 1px -1px ${colour}`,
        "padding-left": "10px",
        "margin-left": "-10px"
    }
}
