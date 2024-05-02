#这里会定义一些过滤补丁的规则，比如过滤掉一些不符合规范的补丁，或者过滤掉一些不符合要求的补丁
#这里的规则是可以自定义的，可以根据实际情况来定义


# 规定补丁的行数补丁大于100行的补丁，大于100行的补丁会被跳过
def filter_patch(patch_content):
    if len(patch_content) > 100:
        return True
    return False