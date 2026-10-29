// Prints every bone's world transform in the setup pose, read with the vendored Spine runtime's public API.
// Usage: node bones.mjs <atlas file> <skeleton .json or .skel>
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const context = vm.createContext({console, TextDecoder, TextEncoder, Float32Array, Uint8Array, Uint16Array, Int16Array,
  Int32Array, ArrayBuffer, DataView});
vm.runInContext(readFileSync(new URL('./vendor/spine-player-4.2.120/spine-player.min.js', import.meta.url), 'utf8'), context);
const S = context.spine;
const [atlasFile, skeletonFile] = process.argv.slice(2);
const atlas = new S.TextureAtlas(readFileSync(atlasFile, 'utf8'));
atlas.setTextures({get: () => ({getImage: () => ({width: 1, height: 1}), setFilters() {}, setWraps() {}})});
const loader = new S.AtlasAttachmentLoader(atlas);
const data = skeletonFile.endsWith('.skel')
  ? new S.SkeletonBinary(loader).readSkeletonData(new Uint8Array(readFileSync(skeletonFile)))
  : new S.SkeletonJson(loader).readSkeletonData(readFileSync(skeletonFile, 'utf8'));
const skeleton = new S.Skeleton(data);
skeleton.setToSetupPose();
skeleton.updateWorldTransform(S.Physics.reset);
process.stdout.write(JSON.stringify(Object.fromEntries(skeleton.bones.map(b => [b.data.name,
  {x: b.worldX, y: b.worldY, a: b.a, b: b.b, c: b.c, d: b.d, parent: b.parent ? b.parent.data.name : null}]))));
