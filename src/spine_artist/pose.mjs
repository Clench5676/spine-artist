// Poses a built skeleton with the vendored Spine runtime's public API (through posing.mjs)
// and prints every pose's geometry as JSON: base, each expression, blink frames and idle frames.
// Usage: node pose.mjs <atlas file> <skeleton .json or .skel> [idleFrames] [blinkFrames]
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {poseTracks, listPoses} from './posing.mjs';
const context = vm.createContext({console, TextDecoder, TextEncoder, Float32Array, Uint8Array, Uint16Array, Int16Array,
  Int32Array, ArrayBuffer, DataView});
vm.runInContext(readFileSync(new URL('./vendor/spine-player-4.2.120/spine-player.min.js', import.meta.url), 'utf8'), context);
const [atlasFile, skeletonFile, idleFrames = '8', blinkFrames = '3'] = process.argv.slice(2);
const atlas = readFileSync(atlasFile, 'utf8');
const skeleton = skeletonFile.endsWith('.skel') ? new Uint8Array(readFileSync(skeletonFile)) : readFileSync(skeletonFile, 'utf8');
const poses = listPoses(context.spine, atlas, skeleton, {idleFrames: +idleFrames, blinkFrames: +blinkFrames});
const animations = (() => {
  if (skeletonFile.endsWith('.skel')) return null;
  return Object.keys(JSON.parse(skeleton).animations ?? {});
})();
process.stdout.write(JSON.stringify({animations, poses: poses.map(p => ({...p, geometry: poseTracks(context.spine, atlas, skeleton, p.tracks)}))}));
