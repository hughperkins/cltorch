-- Copyright Hugh Perkins 2015 hughperkins at gmail
--
-- This Source Code Form is subject to the terms of the Mozilla Public License, 
-- v. 2.0. If a copy of the MPL was not distributed with this file, You can 
-- obtain one at http://mozilla.org/MPL/2.0/.

require 'init'

print('done')

require 'nn'

a = nn.Linear(2,3)
print(a)

a:cl()

