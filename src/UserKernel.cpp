#include "UserKernel.h"

extern "C" {
  #include "lua.h"
  #include "utils.h"
  #include "luaT.h"
}
#include "EasyCL.h"
#include "THClKernels.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
using namespace std;

static std::string getTensorInfoClSrc();

enum ClKernelDirection {
  input,
  output,
  inout
};

class ClKernel;

class ClKernelArg {
public:
  ClKernelDirection direction;
  string name;
  ClKernelArg(ClKernelDirection direction, string name) :
    direction(direction),
    name(name) {
  }
  virtual std::string asParameterString() const = 0;
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) = 0;
  virtual ~ClKernelArg() {
  }
  virtual std::string toString() const {
    return "ClKernelArg{name=" + name + "}";
  }
};

class ClKernelArgInt : public ClKernelArg {
public:
  ClKernelArgInt(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
    if(direction != input) {
      THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string asParameterString() const {
    return "int " + name;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldchecknumber(L, -1, name.c_str());
    int value = lua_tonumber(L, -1);
    lua_pop(L, 1);
    switch(direction) {
      case input:
        k->in(value);
        break;
      default:
        THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string toString() const {
    return "ClKernelArgInt{name=" + name + "}";
  }
};

class ClKernelArgTensor : public ClKernelArg {
public:
  ClKernelArgTensor(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
  }
  virtual std::string asParameterString() const {
    string res = "";
    string conststring = "";
    if(direction == input) {
      conststring = "const";
    }
    res += "global struct THClTensorInfoCl *" + name + "_info, ";
    res += "global " + conststring + " float * " + name + "_data";
    return res;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldcheckudata(L, -1, name.c_str(), "torch.ClTensor");
    THClTensor *value = (THClTensor *)luaT_checkudata(L, -1, "torch.ClTensor");
    if(value == 0) {
      THError("Tensor is null.  this is odd actually, raise an issue");
    }
    if(value->storage == 0) {
      if(direction != output){
        THError("Output tensor has no data");
        return;
      } else {
        // I guess we should resize here, somehow, in the future
        THError("resize not implemented yet.  It should be.  Please remind me to add this, eg raise an issue");
        return;
      }
    }
    if(value->storage->wrapper == 0) {
      THError("resize not implemented yet.  It should be.  Please remind me to add this, eg raise an issue");
      return;
    }
    switch(direction) {
      case input:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->copyToDevice();
        }
        k->inv2(value);
        break;
      case output:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->createOnDevice();
        }
        k->outv2(value);
        break;
      case inout:
        if(!value->storage->wrapper->isOnDevice()) {
          value->storage->wrapper->copyToDevice();
        }
        k->inoutv2(value);
        break;
    }
    lua_pop(L, 1);
  }
  virtual std::string toString() const {
    return "ClKernelArgTensor{name=" + name + "}";
  }
};

class ClKernelArgFloat : public ClKernelArg {
public:
  ClKernelArgFloat(ClKernelDirection direction, string name) :
      ClKernelArg(direction, name) {
    if(direction != input) {
      THError("floats can only be input parameters, not output, or inout");
    }
  }
  virtual std::string asParameterString() const {
    return "float " + name;
  }
  virtual void writeToKernel(lua_State *L, ClKernel *clKernel,THClKernels *k) {
    luaT_getfieldchecknumber(L, -1, name.c_str());
    float value = lua_tonumber(L, -1);
    lua_pop(L, 1);
    switch(direction) {
      case input:
        k->in(value);
        break;
      default:
        THError("ints can only be input parameters, not output, or inout");
    }
  }
  virtual std::string toString() const {
    return "ClKernelArgFloat{name=" + name + "}";
  }
};
int getNumElementsFromTensorArg(lua_State *L, ClKernelArgTensor *arg) {
  THClState *state = cltorch_getstate(L);
  luaT_getfieldcheckudata(L, -1, arg->name.c_str(), "torch.ClTensor");
  THClTensor *tensor = (THClTensor *)luaT_checkudata(L, -1, "torch.ClTensor");
  lua_pop(L, 1);
  return THClTensor_nElement(state, tensor);
}
class ClKernel {
public:
  int refCount;
  string source;
  string extraSource;
  string kernelName;
  string renderedKernel;
  CLKernel *kernel;
  vector< ClKernelArg * >args;
  ClKernel() {
    kernel = 0;
    kernelName = "user_kernel";
    extraSource = "";
  }
  ~ClKernel() {
    for(int i = 0; i < (int)args.size(); i++) {
      delete args[i];
    }
    args.clear();
  }
};
static void ClKernel_rawInit(ClKernel *self) {
  self->refCount = 1;
  self->source = "";
}
static void loadParameters(lua_State *L, ClKernelDirection direction, ClKernel *self) {
  // assume that stack top is a table iwth the parameters
  // each parameter comprises name as key, and type as string value
  // valid typenames: int, float, torch.ClTensor, maybe ClTensor as shorthand
  // iterate this table...
  lua_pushnil(L);
  while(lua_next(L, -2) != 0) {
    string name = lua_tostring(L, -2);
    string paramType = lua_tostring(L, -1);
    if(paramType == "float") {
      ClKernelArg *arg = new ClKernelArgFloat(direction, name);
      self->args.push_back(arg);
    } else if(paramType == "int") {
      ClKernelArg *arg = new ClKernelArgInt(direction, name);
      self->args.push_back(arg);
    } else if(paramType == "torch.ClTensor" || paramType == "ClTensor") {
      ClKernelArg *arg = new ClKernelArgTensor(direction, name);
      self->args.push_back(arg);
    } else {
      THError("Unrecognized typename %s", paramType.c_str());
    }
    lua_pop(L, 1);
  }
}
static int ClKernel_new(lua_State *L) {
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  self = new(self) ClKernel();
  ClKernel_rawInit(self);

  if(lua_type(L, 1) == LUA_TTABLE) {
    lua_pushnil(L);
    while(lua_next(L, -2) != 0) {
      string key = lua_tostring(L, -2);
      if(key == "input" || key == "inputs") {
        loadParameters(L, ClKernelDirection::input, self);
      } else if( key == "output" || key == "outputs") {
        loadParameters(L, ClKernelDirection::output, self);
      } else if( key == "inout" || key == "inouts") {
        loadParameters(L, ClKernelDirection::inout, self);
      } else if( key == "name") {
        self->kernelName = lua_tostring(L, -1); // probably should use luaT for this
      } else if( key == "src") {
        self->source = lua_tostring(L, -1);
      } else if( key == "funcs") {
        self->extraSource = lua_tostring(L, -1);
      } else {
        THError("Parameter %s not recognized", key.c_str());
      }
      lua_pop(L, 1);
    }

    if(self->source == "") {
      THError("Missing parameter src, or was empty");
    }

    THClState *state = cltorch_getstate(L);
    EasyCL *cl = (EasyCL *)THClState_getClv2(state, state->currentDevice);
    string renderedKernel = "";
    renderedKernel += easycl::replaceGlobal(getTensorInfoClSrc(), "{{MAX_CLTORCH_DIMS}}", easycl::toString(MAX_CLTORCH_DIMS)) + "\n";
    renderedKernel += self->extraSource + "\n";
    renderedKernel += "kernel void " + self->kernelName + "(\n";
    for(int i = 0; i < (int)self->args.size(); i++) {
      if(i > 0) {
        renderedKernel += ",\n";
      }
      renderedKernel += "    " + self->args[i]->asParameterString();
    }
    renderedKernel += "\n) {\n";   // probalby should use ostringstream for this really, for speed
    renderedKernel += self->source + "\n";
    renderedKernel += "}\n";
    self->renderedKernel = renderedKernel;
    try {
      self->kernel = cl->buildKernelFromString(renderedKernel, self->kernelName, "", "UserKernel");
    } catch(runtime_error &e) {
      THError("Failed to create kernel %s", e.what());
    }
  } else {
    THError("First parameter to torch.ClKernel should be a table");
  }

  luaT_pushudata(L, self, "torch.ClKernel");
  return 1;
}
static int ClKernel_free(lua_State *L) {
  ClKernel *self = (ClKernel*)THAlloc(sizeof(ClKernel));
  if(!self) {
    return 0;
  }
  if(THAtomicDecrementRef(&self->refCount))
  {
    delete self->kernel;
    self->~ClKernel();
    THFree(self);
  }
  return 0;
}
static int ClKernel_factory(lua_State *L) {
  THError("ClKernel_factory not implemented");
  return 0;
}
static int ClKernel_print(lua_State *L) {
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  cout << "Original source\n"
       << "===============\n"
       << self->source << endl;
  cout << endl;
  cout << "Generated source\n"
       << "================\n"
       << self->renderedKernel << endl;
  return 0;
}
static int ClKernel_getRawKernel(lua_State *L) {
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  lua_pushstring(L, self->source.c_str());
  return 1;
}
static int ClKernel_getRenderedKernel(lua_State *L) {
  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  lua_pushstring(L, self->renderedKernel.c_str());
  return 1;
}
static int ClKernel_run(lua_State *L) {
  THClState *state = cltorch_getstate(L);
  if(lua_type(L, 2) != LUA_TTABLE) {
    THError("usage  :run(inputParamsTable [,optionsTable]");
    return 0;
  }
  // now we can assume we have a table :-)

  ClKernel *self = (ClKernel *)luaT_checkudata(L, 1, "torch.ClKernel");
  THClKernels k(state, self->kernel);
  
  lua_pushvalue(L, 2);
  int numElements = -1;
  for(int i = 0; i < (int)self->args.size(); i++) {
    // what we need to do here is:
    // - loop through each parameter object
    // - for each parameter object, get the value from the passed in parameters, which
    //   were hopefully passed into this method, as values in a table
    // - and then write this value to the kernel
    // presumably, we can get each arg to extract itself from the table, as long as
    // the table is top of the stack
    // we can ignore extra values in the table for now
    // what we do is, throw error on missing values
    self->args[i]->writeToKernel(L, self, &k);
    if(numElements == -1 && self->args[i]->direction != input && dynamic_cast< ClKernelArgTensor *>( self->args[i] ) != 0) {
      numElements = getNumElementsFromTensorArg(L, dynamic_cast< ClKernelArgTensor *>(self->args[i]));
    }
  }
  if(numElements == -1) {
    THError("Must provide at least one output, or inout, ClTensor");
  }
  lua_pop(L, 1);

  // process any options
  // any unrecognized option => error
  int workgroupSize = 64;
  int numWorkgroups = (numElements + workgroupSize - 1) / workgroupSize;
  if(lua_gettop(L) >= 3) {
    if(lua_type(L, 3) != LUA_TTABLE) {
      THError("usage  :run(inputParamsTable [,optionsTable]");
      return 0;
    }
    lua_pushnil(L);
    while(lua_next(L, -2) != 0) {
      if(!lua_isstring(L, -2)) {
        THError("Options keys should be strings");
      }
      string key = lua_tostring(L, -2);
      if(key == "workgroupSize") {
        if(!lua_isnumber(L, -1) ) {
          THError("workgroupSize should be an integer");
        }
        workgroupSize = lua_tonumber(L, -1);
      } else if(key == "numWorkgroups") {
        if(!lua_isnumber(L, -1) ) {
          THError("numWorkgroups should be an integer");
        }
        numWorkgroups = lua_tonumber(L, -1);
      } else {
        THError("option %s not recognized", key.c_str());
      }
      lua_pop(L, 1);
    }
  }
  int globalSize = workgroupSize * numWorkgroups;
  self->kernel->run_1d(globalSize, workgroupSize);
  return 0;
}
static const struct luaL_Reg ClKernel_funcs [] = {
  {"print", ClKernel_print},
  {"getRenderedKernel", ClKernel_getRenderedKernel},
  {"getRawKernel", ClKernel_getRawKernel},
  {"run", ClKernel_run},
  {0,0}
};
void cltorch_UserKernel_init(lua_State *L)
{
  luaT_newmetatable(L, "torch.ClKernel", NULL,
                    ClKernel_new, ClKernel_free, ClKernel_factory);
  luaL_setfuncs(L, ClKernel_funcs, 0);
  lua_pop(L, 1);
}
static std::string getTensorInfoClSrc() {
  // [[[cog
  // import stringify
  // stringify.write_kernel( "kernel", "src/lib/THClTensorInfoCl.cl" )
  // ]]]
  // generated using cog, from src/lib/THClTensorInfoCl.cl:
  const char * kernelSource =  
  "typedef struct THClTensorInfoCl {\n" 
  "  unsigned int sizes[{{MAX_CLTORCH_DIMS}}];\n" 
  "  unsigned int strides[{{MAX_CLTORCH_DIMS}}];\n" 
  "  int offset;\n" 
  "  int dims;\n" 
  "} TensorInfoCl;\n" 
  "\n" 
  "";
  // [[[end]]]
  return kernelSource;
}

