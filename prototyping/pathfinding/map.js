

export class Map{

	constructor(width, height){
		this.width = width;
		this.height = height;
		this.cells = new Array(this.width * this.height).fill(0);
		this.visited = new Array(this.width * this.height).fill(false);
		this.distances = new Array(this.width * this.height).fill(Infinity);
		this.previous = new Array(this.width * this.height).fill(undefined);

		this.cellSize = 30;
	}

	createSvg(){

		let node = document.createElementNS('http://www.w3.org/2000/svg','g');

		

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
			
			let u = this.cellSize * x;
			let v = this.cellSize * y;

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

	pathToSvg(path){

		let node = document.createElementNS('http://www.w3.org/2000/svg','g');

		for(let i = 0; i < path.length - 1; i++){

			let start = path[i];
			let end = path[i+1];

			let x1 = this.cellSize * start.x + this.cellSize * 0.5;
			let y1 = this.cellSize * start.y + this.cellSize * 0.5;
			let x2 = this.cellSize * end.x + this.cellSize * 0.5;
			let y2 = this.cellSize * end.y + this.cellSize * 0.5;

			let line = document.createElementNS('http://www.w3.org/2000/svg','line');
			line.setAttribute('x1', `${x1}`);
			line.setAttribute('y1', `${y1}`);
			line.setAttribute('x2', `${x2}`);
			line.setAttribute('y2', `${y2}`);
			line.setAttribute('stroke', 'red');

			if(i === path.length - 2){
				line.setAttribute('marker-end', 'url(#arrow)');
			}

			node.appendChild(line);

			if(i === 0){
				let radius = 0.3 * this.cellSize;
				let circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
				circle.setAttribute('cx', `${x1}`);
				circle.setAttribute('cy', `${y1}`);
				circle.setAttribute('r', `${radius}`);
				circle.setAttribute('fill', 'red');
				node.appendChild(circle);
			}
			

		}

		return node;
	}

	toCellID(cx, cy){
		return cx + this.width * cy;
	}

	fromCellID(cellID){
		return {
			x: cellID % this.width,
			y: Math.floor(cellID / this.width)
		};
	}

	// findPathAStar(start, end){

	// }

	findPathDijkstra(start, end, enableTrace){

		let t_start = Date.now();

		this.distances.fill(Infinity);
		this.visited.fill(false);
		this.previous.fill(undefined);

		let targetCellID = this.toCellID(...end);
		let startCellID = this.toCellID(...start);
		
		// Initialize with current cell = start
		let currentCell = {x: start[0], y: start[1]};
		let currentCellID = this.toCellID(currentCell.x, currentCell.y);
		this.distances[currentCellID] = 0;

		let path = [];
		let numNeighborChecks = 0;
		
		let abc = 0;
		outerLoop:
		while(abc < 10_000){
			abc++;

			// mark current cell as visited
			this.visited[currentCellID] = true;

			path.push(currentCell);

			// CHECK/UPDATE NEIGHBORS
			neigborLoop:
			for(let dx = -1; dx <= 1; dx++)
			for(let dy = -1; dy <= 1; dy++)
			{
				let neighbor_x = currentCell.x + dx;
				let neighbor_y = currentCell.y + dy;
				let neighbor_id = this.toCellID(neighbor_x, neighbor_y);

				if(neighbor_id == currentCellID) continue;

				numNeighborChecks++;
				
				// skip if 
				if(neighbor_x < 0 || neighbor_x >= this.width) continue;   // outside map
				if(neighbor_y < 0 || neighbor_y >= this.height) continue;
				if(this.visited[neighbor_id]) continue;                    // neighbor was visited

				// compute distance from current to neighbor
				let weight = Math.max(this.cells[neighbor_id], 1);
				let stepDistance = weight * Math.sqrt(dx * dx + dy * dy);
				let neighborDistance = this.distances[currentCellID] + stepDistance;

				// if that distance is smaller than previously stored distance, replace with new one
				if(neighborDistance < this.distances[neighbor_id]){
					this.distances[neighbor_id] = neighborDistance;
					this.previous[neighbor_id] = currentCellID;
				}

			}

			//now find cell with smallest travel distance
			let i_smallest = 0;
			let d_smallest = Infinity;
			for(let i = 0; i < this.width * this.height; i++){
				if(this.distances[i] < d_smallest && !this.visited[i]){
					i_smallest = i;
					d_smallest = this.distances[i];
				}
			}

			if(d_smallest == Infinity){
				console.log("done?");
				break;
			}else{
				currentCell = {
					x: i_smallest % this.width,
					y: Math.floor(i_smallest / this.width)
				}
				currentCellID = i_smallest;

				if(currentCellID == targetCellID){

					path = [{x: end[0], y: end[1]}];
					for(let i = 0; i < 10_000; i++){

						let previousID = this.previous[currentCellID];
						let previous = this.fromCellID(previousID);

						path.push(previous);

						if(previousID == startCellID){
							path.reverse();
							break;
						} 

						currentCellID = previousID;
					}

					path.push({x: end[0], y: end[1]});

					let t_end = Date.now();
					let millies = t_end - t_start;
					// console.log(`milliseconds: ${millies} ms`);
					break outerLoop;
					// return path;
				}
			}

		}

		

		return {path, numNeighborChecks};

	}

};

