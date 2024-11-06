window.finish = false;
var img = $("body > table > tbody > tr > td > table:nth-child(6) > tbody > tr > td > form > table:nth-child(7) > tbody > tr > td:nth-child(1) > table > tbody > tr:nth-child(1) > td img");
var link = $('<a>');
link.attr('href', img.attr('src'));	    
link.attr('download', "1".toString().padStart(5, '0') + "_M.jpg");
$('body').append(link);
link[0].click();
link.remove();
var images = $("body > table > tbody > tr > td > table:nth-child(6) > tbody > tr > td > form > div:nth-child(12) img");
var totalRequests = images.length;
var completedRequests = 0;
$.each(images, function(i, image) {
var imageUrl = $(image).attr('src');
	if (!imageUrl.endsWith('.gif')) {
		setTimeout(function() {
			var link1 = $('<a>');
			link1.attr('href', $(image).attr('src'));	    
			link1.attr('download', "1".toString().padStart(5, '0') + "_" + (i + 1).toString().padStart(2, '0') + "." + $(image).attr('src').slice(($(image).attr('src').lastIndexOf(".") - 1 >>> 0) + 2));
			$('body').append(link1);
			link1[0].click();
			link1.remove();
			completedRequests++;
			finish = totalRequests == completedRequests;
		}, i * 2000);
	} else {
		totalRequests--;
	}
});
