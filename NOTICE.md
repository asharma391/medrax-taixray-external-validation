# Attribution and scope

This is an independent, data-free audit. It is not affiliated with, endorsed
by, or a distribution of MedRAX, Wang Lab, TAIX-Ray, or their institutions.

The evaluation adapter deliberately mirrors the public configuration of
`bowang-lab/MedRAX`'s `ChestXRayClassifierTool` at commit
`d243cc50348fa0f0c219cc623e2cf26b906be676`: the
`torchxrayvision` `densenet121-res224-all` weight identifier, XRV
normalization, and center crop. The upstream project is Apache-2.0 licensed.
The compatibility patch is published separately as a transparent fork at
https://github.com/asharma391/MedRAX.

TAIX-Ray is used under CC-BY-4.0. This repository does not redistribute its
images, patient/physician identifiers, signed image URLs, or row-level labels.
Use the official dataset and follow its data-use notice and applicable law.
