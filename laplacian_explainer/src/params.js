"use strict";
//=================== ПАРАМЕТРЫ (как в casynth_core) ===================
const SIZE = 8;                 // PATCH_SIZE
const SR = 44100;
const GUARD = 0.45 * SR;        // anti-alias guard
const NEIGH = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]]; // 8-связность

//=================== СОСТОЯНИЕ ===================
let cells = Array.from({length:SIZE},()=>Array(SIZE).fill(0));
let selMode = -1;               // выбранная мода для визуализации (индекс в полном спектре λ)
let last = null;                // последний результат вычисления
let modeMask = [];              // boolean[], длина = modeFreqs.length
let maskDirty = true;           // перестроить маску при следующем recompute

const $ = id => document.getElementById(id);
