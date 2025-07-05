use crate::client::EncryptedGrid;
use tfhe::prelude::*;
use tfhe::{FheUint4, ServerKey, set_server_key};
use rayon::prelude::*;

pub(crate) struct Server {
    server_key: ServerKey,
    grid: EncryptedGrid,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        Server { server_key, grid }
    }

    pub(crate) fn run(&self, steps: u32) -> EncryptedGrid {
        rayon::broadcast(|_| set_server_key(self.server_key.clone()));
        set_server_key(self.server_key.clone());
        
        let mut current_grid = self.grid.clone();
        for step in 0..steps {
            println!("Running step {}/{}", step + 1, steps);
            current_grid = self.step(&current_grid);
        }
        current_grid
    }

    fn step(&self, grid: &EncryptedGrid) -> EncryptedGrid {
        (0..grid.len())
            .into_par_iter()
            .map(|i| {
                (0..grid[i].len())
                    .into_par_iter()
                    .map(|j| {self.update_cell(i, j, grid)})
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn update_cell(&self, x: usize, y: usize, grid: &EncryptedGrid) -> FheUint4 {
        let neighbors: Vec<(isize, isize)> = vec![
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ];

        let count = neighbors
            .into_par_iter()
            .filter_map(|(dx, dy)| {
                let nx = x.wrapping_add(dx as usize);
                let ny = y.wrapping_add(dy as usize);
                
                if nx < grid.len() && ny < grid[nx].len() {
                    Some(grid[nx][ny].clone())
                } else {
                    None
                }
            })
            .reduce(
                || FheUint4::try_encrypt_trivial(0u8).unwrap(),
                |acc, val| acc + val
            );

        let cell = &grid[x][y];
        let zero = FheUint4::try_encrypt_trivial(0u8).unwrap();
        let one = FheUint4::try_encrypt_trivial(1u8).unwrap();
        let two = FheUint4::try_encrypt_trivial(2u8).unwrap();
        let three = FheUint4::try_encrypt_trivial(3u8).unwrap();

        let alive = cell.eq(&one);
        let eq_three = count.eq(&three).select(&one, &zero);
        let eq_two_or_three = (count.eq(&two) | count.eq(&three)).select(&one, &zero);

        alive.if_then_else(&eq_two_or_three, &eq_three)
    }
}
