// Papers whose results were obtained with RaiSim, for the home page's Hall of Fame
// and the Hall of Fame page. `evidence` records where the paper (or its authors'
// institution) says RaiSim was used; keep it when adding an entry.

export interface Paper {
  title: string
  /** Short name and venue for the home page's list of other papers. */
  short?: string
  venueShort?: string
  authors: string[]
  /** Where the work was done, as shown on the cards. */
  institutions: string
  venue: string
  year: number
  /** Volume, issue and article or pages, as cited. */
  citation?: string
  url: string
  /** One line about the result, from the paper's abstract. */
  highlight: string
  evidence: string
  kind: 'nature' | 'science-robotics' | 'other'
}

export const papers: Paper[] = [
  {
    title: 'A quadruped robot designed to complete a marathon on a single battery charge',
    authors: ['C. Lee', 'D. Youm', 'J. Park', 'J. Lee', 'S. Choi', 'G. Ji', 'J. Mun', 'M. Jung', 'H. C. Choi', 'H. Kim', 'H. Oh', 'K. Nam', 'M. Lee', 'J. Hur', 'D. Choi', 'D. Kim', 'Y. An', 'J. Hwangbo'],
    institutions: 'KAIST',
    venue: 'Nature',
    year: 2026,
    citation: '658(8135), 372–379',
    url: 'https://doi.org/10.1038/s41586-026-11102-5',
    highlight: 'RAIBO2 ran a full marathon on one battery charge, in 4 hours 19 minutes.',
    evidence: 'KAIST news, 2024-11-15 and 2024-11-17: the walking controller was trained "using their proprietary dynamic simulator RaiSim".',
    kind: 'nature',
  },
  {
    title: 'High-speed control and navigation for quadrupedal robots on complex and discrete terrain',
    authors: ['H. Kim', 'H. Oh', 'J. Park', 'Y. Kim', 'D. Youm', 'M. Jung', 'M. Lee', 'J. Hwangbo'],
    institutions: 'KAIST',
    venue: 'Science Robotics',
    year: 2025,
    citation: '10(102), eads6192',
    url: 'https://doi.org/10.1126/scirobotics.ads6192',
    highlight: 'Raibo runs on vertical walls, jumps a 1.3 m gap and crosses stepping stones at 4 m/s.',
    evidence: 'arXiv:2506.02835, Methods: "We generated 300 environments in parallel using the Raisim physics simulation".',
    kind: 'science-robotics',
  },
  {
    title: 'Learning quadrupedal locomotion on deformable terrain',
    authors: ['S. Choi', 'G. Ji', 'J. Park', 'H. Kim', 'J. Mun', 'J. H. Lee', 'J. Hwangbo'],
    institutions: 'KAIST',
    venue: 'Science Robotics',
    year: 2023,
    citation: '8(74), eade2256',
    url: 'https://doi.org/10.1126/scirobotics.ade2256',
    highlight: 'Raibo runs on soft beach sand at 3.03 m/s, its feet buried at every step.',
    evidence: 'Listed by the RaiSim authors (2026-10-25); the paper is not openly available to quote.',
    kind: 'science-robotics',
  },
  {
    title: 'Learning robust perceptive locomotion for quadrupedal robots in the wild',
    authors: ['T. Miki', 'J. Lee', 'J. Hwangbo', 'L. Wellhausen', 'V. Koltun', 'M. Hutter'],
    institutions: 'ETH Zurich, KAIST and Intel',
    venue: 'Science Robotics',
    year: 2022,
    citation: '7(62), eabk2822',
    url: 'https://doi.org/10.1126/scirobotics.abk2822',
    highlight: 'ANYmal fuses vision and proprioception with zero falls in alpine, forest, underground and urban terrain.',
    evidence: 'arXiv:2201.08117, Methods: "We use RaiSim as our simulator to build the training environment."',
    kind: 'science-robotics',
  },
  {
    title: 'Learning quadrupedal locomotion over challenging terrain',
    authors: ['J. Lee', 'J. Hwangbo', 'L. Wellhausen', 'V. Koltun', 'M. Hutter'],
    institutions: 'ETH Zurich, KAIST and Intel',
    venue: 'Science Robotics',
    year: 2020,
    citation: '5(47), eabc5986',
    url: 'https://doi.org/10.1126/scirobotics.abc5986',
    highlight: 'Without vision, ANYmal trots through mud, sand, rubble, thick vegetation, snow and running water.',
    evidence: 'arXiv:2010.11251, Methods: "The RaiSim simulator is used for rigid-body and contact dynamics simulation."',
    kind: 'science-robotics',
  },
  {
    title: 'Learning agile and dynamic motor skills for legged robots',
    authors: ['J. Hwangbo', 'J. Lee', 'A. Dosovitskiy', 'D. Bellicoso', 'V. Tsounis', 'V. Koltun', 'M. Hutter'],
    institutions: 'ETH Zurich and Intel',
    venue: 'Science Robotics',
    year: 2019,
    citation: '4(26), eaau5872',
    url: 'https://doi.org/10.1126/scirobotics.aau5872',
    highlight: 'ANYmal runs faster than ever before and recovers from falls, with policies trained in simulation.',
    evidence: 'arXiv:1901.08652: trained in the authors\' in-house simulator built on the per-contact iteration contact solver (RA-L 2018, www.raisim.com), the first version of RaiSim.',
    kind: 'science-robotics',
  },
  {
    title: 'RMA: Rapid Motor Adaptation for Legged Robots',
    short: 'RMA: Rapid Motor Adaptation',
    venueShort: 'RSS',
    authors: ['A. Kumar', 'Z. Fu', 'D. Pathak', 'J. Malik'],
    institutions: 'UC Berkeley, CMU and Meta AI',
    venue: 'Robotics: Science and Systems',
    year: 2021,
    citation: 'XVII, 011',
    url: 'https://doi.org/10.15607/RSS.2021.XVII.011',
    highlight: 'An A1 adapts to new terrain in a fraction of a second, with no fine-tuning on the robot.',
    evidence: 'arXiv:2107.04034, Simulation Setup: "We use the RaiSim simulator".',
    kind: 'other',
  },
  {
    title: 'Concurrent training of a control policy and a state estimator for dynamic and robust legged locomotion',
    short: 'Concurrent training of a policy and a state estimator',
    venueShort: 'RA-L',
    authors: ['G. Ji', 'J. Mun', 'H. Kim', 'J. Hwangbo'],
    institutions: 'KAIST',
    venue: 'IEEE Robotics and Automation Letters',
    year: 2022,
    citation: '7(2)',
    url: 'https://doi.org/10.1109/LRA.2022.3151396',
    highlight: 'The policy and its state estimator learn together, for dynamic and robust locomotion.',
    evidence: 'arXiv:2202.05481: "trained in simulation using RaiSim".',
    kind: 'other',
  },
  {
    title: 'Circus ANYmal: A quadruped learning dexterous manipulation with its limbs',
    short: 'Circus ANYmal: dexterous manipulation with its limbs',
    venueShort: 'ICRA',
    authors: ['F. Shi', 'T. Homberger', 'J. Lee', 'T. Miki', 'M. Zhao', 'F. Farshidian', 'K. Okada', 'M. Inaba', 'M. Hutter'],
    institutions: 'The University of Tokyo and ETH Zurich',
    venue: 'IEEE International Conference on Robotics and Automation',
    year: 2021,
    citation: 'pp. 2316–2323',
    url: 'https://arxiv.org/abs/2011.08811',
    highlight: 'ANYmal balances and manipulates a light ball with its limbs, transferred zero-shot from simulation.',
    evidence: 'arXiv:2011.08811: "The policy is trained in the Raisim simulator".',
    kind: 'other',
  },
  {
    title: 'DeepGait: Planning and control of quadrupedal gaits using deep reinforcement learning',
    short: 'DeepGait: gait planning and control',
    venueShort: 'RA-L',
    authors: ['V. Tsounis', 'M. Alge', 'J. Lee', 'F. Farshidian', 'M. Hutter'],
    institutions: 'ETH Zurich',
    venue: 'IEEE Robotics and Automation Letters',
    year: 2020,
    url: 'https://arxiv.org/abs/1909.08399',
    highlight: 'A learned gait planner and controller cross narrow bridges, gaps and stepping stones in simulation.',
    evidence: 'arXiv:1909.08399: "we used the RaiSim multi-body physics engine".',
    kind: 'other',
  },
]
