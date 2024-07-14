

export class Map{

	constructor(){
		this.width = 16;
		this.height = 16;
		this.cells = new Array(this.width * this.height).fill(0);
	}

	createSvg(){

		let node = document.createElementNS('http://www.w3.org/2000/svg','g');

		let cellSize = 30;

		// { // border
		// 	let rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
		// 	rect.setAttribute('x', `0`);
		// 	rect.setAttribute('y', `0`);
		// 	rect.setAttribute('width',  `${cellSize * this.width}`);
		// 	rect.setAttribute('height', `${cellSize * this.height}`);
		// 	rect.setAttribute('stroke', 'black');
			
		// 	node.appendChild(rect);
		// }

		for(let x = 0; x < this.width; x++)
		for(let y = 0; y < this.height; y++)
		{
			
			let u = cellSize * x;
			let v = cellSize * y;

			// let colorState = (x + y) % 2;
			let cellID = x + y * this.width;
			let colorState = this.cells[cellID];
			let color = colorState === 0 ? "white" : "black";

			let rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
			rect.setAttribute('x', `${u}`);
			rect.setAttribute('y', `${v}`);
			rect.setAttribute('width', '30');
			rect.setAttribute('height', '30');
			rect.setAttribute('fill', color);
			rect.setAttribute('stroke', 'black');
			
			node.appendChild(rect);
		}


		return node;
	}

};

