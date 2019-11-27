# Group calls

export groupStart, groupEnd, group

groupStart() = ncclGroupStart()
groupEnd()   = ncclGroupEnd()

function group(f)
    groupStart()
    f()
    groupEnd()
end
