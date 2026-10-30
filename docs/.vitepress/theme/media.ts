// Home-page media. The hero reel and most tiles use small H.264 encodes of
// RaiSim simulations rendered with rayrai (see rsc/docs/video/home/README.md).
// Physics cards also reuse example-gallery clips and their posters.
import heroOffroad from '../../../rsc/docs/video/home/hero_offroad.mp4'
import heroOffroadSm from '../../../rsc/docs/video/home/hero_offroad-960.mp4'
import heroOffroadPoster from '../../../rsc/docs/video/home/hero_offroad.jpg'
import heroForest from '../../../rsc/docs/video/home/hero_forest.mp4'
import heroForestSm from '../../../rsc/docs/video/home/hero_forest-960.mp4'
import heroForestPoster from '../../../rsc/docs/video/home/hero_forest.jpg'
import heroCity from '../../../rsc/docs/video/home/hero_city.mp4'
import heroCitySm from '../../../rsc/docs/video/home/hero_city-960.mp4'
import heroCityPoster from '../../../rsc/docs/video/home/hero_city.jpg'
import heroRain from '../../../rsc/docs/video/home/hero_rain.mp4'
import heroRainSm from '../../../rsc/docs/video/home/hero_rain-960.mp4'
import heroRainPoster from '../../../rsc/docs/video/home/hero_rain.jpg'

import glass from '../../../rsc/docs/video/home/tile_glass.mp4'
import glassPoster from '../../../rsc/docs/video/home/tile_glass.jpg'
import lidar from '../../../rsc/docs/video/home/tile_lidar.mp4'
import lidarPoster from '../../../rsc/docs/video/home/tile_lidar.jpg'
import depth from '../../../rsc/docs/video/home/tile_depth.mp4'
import depthPoster from '../../../rsc/docs/video/home/tile_depth.jpg'
import rgb from '../../../rsc/docs/video/home/tile_rgb.mp4'
import rgbPoster from '../../../rsc/docs/video/home/tile_rgb.jpg'
import soil from '../../../rsc/docs/video/home/tile_soil.mp4'
import soilPoster from '../../../rsc/docs/video/home/tile_soil.jpg'
import strandbeest from '../../../rsc/docs/video/home/tile_strandbeest.mp4'
import strandbeestPoster from '../../../rsc/docs/video/home/tile_strandbeest.jpg'
import hand from '../../../rsc/docs/video/home/tile_hand.mp4'
import handPoster from '../../../rsc/docs/video/home/tile_hand.jpg'
import usd from '../../../rsc/docs/video/home/tile_usd.mp4'
import usdPoster from '../../../rsc/docs/video/home/tile_usd.jpg'
import army from '../../../rsc/docs/video/home/tile_army.mp4'
import armyPoster from '../../../rsc/docs/video/home/tile_army.jpg'
import offroadDrive from '../../../rsc/docs/video/home/tile_offroad_drive.mp4'
import offroadDrivePoster from '../../../rsc/docs/video/home/tile_offroad_drive.jpg'
import tracks from '../../../rsc/docs/video/home/tile_tracks.mp4'
import tracksPoster from '../../../rsc/docs/video/home/tile_tracks.jpg'
import gymPolicy from '../../../rsc/docs/video/home/tile_gym.mp4'
import gymPolicyPoster from '../../../rsc/docs/video/home/tile_gym.jpg'
import fluids from '../../../rsc/docs/video/examples/underwater_rov.mp4'
import fluidsPoster from '../../../rsc/docs/video/examples/underwater_rov.jpg'
import deformables from '../../../rsc/docs/video/examples/deformable_objects.mp4'
import deformablesPoster from '../../../rsc/docs/video/examples/deformable_objects.jpg'
import tendons from '../../../rsc/docs/video/home/tile_tendons.mp4'
import tendonsPoster from '../../../rsc/docs/video/home/tile_tendons.jpg'
import tcpViewer from '../../../rsc/docs/video/home/tcp_viewer.jpg'

export interface Clip {
  src: string
  poster: string
}

export interface HeroClip extends Clip {
  srcSmall: string
  title: string
  caption: string
}

export const heroClips: HeroClip[] = [
  {
    src: heroOffroad,
    srcSmall: heroOffroadSm,
    poster: heroOffroadPoster,
    title: 'Off-road',
    caption: 'An off-road buggy with articulated suspension leaves the start line',
  },
  {
    src: heroForest,
    srcSmall: heroForestSm,
    poster: heroForestPoster,
    title: 'Forest',
    caption: 'ANYmal walks through dense, wind-blown forest foliage',
  },
  {
    src: heroCity,
    srcSmall: heroCitySm,
    poster: heroCityPoster,
    title: 'City',
    caption: 'Quadrupeds and wheeled robots share a city crosswalk',
  },
  {
    src: heroRain,
    srcSmall: heroRainSm,
    poster: heroRainPoster,
    title: 'Weather',
    caption: 'Rain, wet asphalt and a robot crossing the street',
  },
]

export const clips = {
  forest: { src: heroForestSm, poster: heroForestPoster },
  city: { src: heroCitySm, poster: heroCityPoster },
  rain: { src: heroRainSm, poster: heroRainPoster },
  glass: { src: glass, poster: glassPoster },
  lidar: { src: lidar, poster: lidarPoster },
  depth: { src: depth, poster: depthPoster },
  rgb: { src: rgb, poster: rgbPoster },
  fluids: { src: fluids, poster: fluidsPoster },
  deformables: { src: deformables, poster: deformablesPoster },
  tendons: { src: tendons, poster: tendonsPoster },
  soil: { src: soil, poster: soilPoster },
  strandbeest: { src: strandbeest, poster: strandbeestPoster },
  hand: { src: hand, poster: handPoster },
  usd: { src: usd, poster: usdPoster },
  army: { src: army, poster: armyPoster },
  offroadDrive: { src: offroadDrive, poster: offroadDrivePoster },
  tracks: { src: tracks, poster: tracksPoster },
  gymPolicy: { src: gymPolicy, poster: gymPolicyPoster },
} satisfies Record<string, Clip>

export const images = { tcpViewer }
